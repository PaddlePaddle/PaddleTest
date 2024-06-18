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



class PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_07e622a39f9ba29f914053559426dc46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cbb5bcc1f495bd5eaaad60809a520f00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_df2f90baf999917130b1ee818f16971c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a7575c03aa1dedfe5932d82113cc07d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 23, 35], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.058322638273239136], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5a2de620b6d5b8dfa2bd8fe89555a496(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_fbb9c2a55e29b92b6c73c6944766cead(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 136, 136], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0decf1ab0797f1ebd880929ceedaa96e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_aade929e4e268bbada6dd5c068e93e29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 18, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_59945b092cb4d746030eeb6abfa865f5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5c0e0fa8e7060cd7f2dbeeb52c0436a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5c0e0fa8e7060cd7f2dbeeb52c0436a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5c0e0fa8e7060cd7f2dbeeb52c0436a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5c0e0fa8e7060cd7f2dbeeb52c0436a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5c0e0fa8e7060cd7f2dbeeb52c0436a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5f5d1ee7a72b78cbc78ef6efd8749dda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_89bebc66b0901f62b0887b762b4ac201(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 19, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6867c07f499cbb344d0274143c66f195(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_627c2ad0e3dd6aa449cd848decc9c1e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[-0.698381781578064, -0.8055073618888855]], [[0.36464864015579224, 0.8828316926956177]], [[-0.7775907516479492, -0.4177951216697693]], [[0.24364709854125977, -0.5350531339645386]], [[0.18365156650543213, 0.4492831230163574]], [[0.15637677907943726, -0.38907161355018616]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


class TestPrimitiveOp_518ba5918f813c447fdf6626e05027fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[0.09632712602615356, -0.08038312196731567]], [[0.8459548354148865, 0.946026086807251]], [[-0.5461992025375366, -0.4563255310058594]], [[-0.1192585825920105, -0.621944785118103]], [[0.27789366245269775, -0.01198965311050415]], [[0.5104289054870605, -0.5729514360427856]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


class TestPrimitiveOp_8acbe515969ee2a95a9551b5ddbbc706(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.698381781578064, -0.8055073618888855]], [[0.36464864015579224, 0.8828316926956177]], [[-0.7775907516479492, -0.4177951216697693]], [[0.24364709854125977, -0.5350531339645386]], [[0.18365156650543213, 0.4492831230163574]], [[0.15637677907943726, -0.38907161355018616]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[-0.698381781578064, -0.8055073618888855]], [[0.36464864015579224, 0.8828316926956177]], [[-0.7775907516479492, -0.4177951216697693]], [[0.24364709854125977, -0.5350531339645386]], [[0.18365156650543213, 0.4492831230163574]], [[0.15637677907943726, -0.38907161355018616]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


class TestPrimitiveOp_b37989828e9d1a5182df04113930386c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.09632712602615356, -0.08038312196731567]], [[0.8459548354148865, 0.946026086807251]], [[-0.5461992025375366, -0.4563255310058594]], [[-0.1192585825920105, -0.621944785118103]], [[0.27789366245269775, -0.01198965311050415]], [[0.5104289054870605, -0.5729514360427856]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.09632712602615356, -0.08038312196731567]], [[0.8459548354148865, 0.946026086807251]], [[-0.5461992025375366, -0.4563255310058594]], [[-0.1192585825920105, -0.621944785118103]], [[0.27789366245269775, -0.01198965311050415]], [[0.5104289054870605, -0.5729514360427856]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


class PrimitiveOp_15e9618881bbcf2704b6c3e42456f235(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1d1a3443e0c693961b81baa0c952d308(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.to_tensor([[[1.2117127180099487], [0.8714643716812134], [0.6878179907798767], [0.20321078598499298], [0.11434483528137207], [0.07372946292161942]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([[[0.27302616834640503], [-0.14556801319122314], [-0.12867471575737], [-0.20281419157981873], [-0.20082539319992065], [0.1963663101196289]]], dtype='float32').reshape([1, 6, 1]),
        ]


class TestPrimitiveOp_b54408eb871163a83a5222ca485d99f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.001974795013666153], [2.0440125465393066], [0.36054110527038574], [0.2539675235748291], [0.021520258858799934], [0.45181843638420105]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([[[0.27302616834640503], [-0.14556801319122314], [-0.12867471575737], [-0.20281419157981873], [-0.20082539319992065], [0.1963663101196289]]], dtype='float32').reshape([1, 6, 1]),
        ]


class TestPrimitiveOp_ae723d1f4e2a0af986110d50ff3a1e16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d66569920e5031c31a8c849401aa1471(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 100, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5f566e56ae9df223c4d0d17844ec0901(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_476e6620e1438088f38ad912418c2905(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d2dbd950f57b82164c50cebf12ebd7ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([100, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d2dbd950f57b82164c50cebf12ebd7ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([100, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_28e70d6740dc6404c75ba5563a0d8aca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0d68d954a3469da5a1a868f34a4b414d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 22, 22], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5a2de620b6d5b8dfa2bd8fe89555a496(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8e63beda2dfc66b4d555c62861281bf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_328bbe04eae11810a126cd9a46ac9e18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 76, 116], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.2697003483772278], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0decf1ab0797f1ebd880929ceedaa96e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_213b4bf75671aeb4c313fe012ba2bb0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 44, 44], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f085ee5c64cb5ce32e2bf1869ae84561(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1025, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e73053575bc7122acb3845762885fe17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1025, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e45b7328d537e9cb5a6ba767e4c5dc3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d8e4ad5a80f96102a6ffa5021229f736(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([22, 60, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 60, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7ccecca78825af7acba6e9bfd323581b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0e66f5bd2c5458c016b3f9caef8ac217(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_eeffab07eb155f1b47f02a756c5d8da5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 13, 13], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_74d80e711c6803921d436e24fd81100a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([15200, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([15200, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_74d80e711c6803921d436e24fd81100a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([15200, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([15200, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_74d80e711c6803921d436e24fd81100a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([15200, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([15200, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_182e5be9a78d93d86dc49cf8d001202a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ca46b07e718270123aca4e2082603295(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_182e5be9a78d93d86dc49cf8d001202a
    def get_inputs(self):
        return [
            paddle.to_tensor([1524], dtype='int32').reshape([1]),
            paddle.to_tensor([4], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_ec2f4ccfb3412b2eb4ecbbe9a330f092(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ec2f4ccfb3412b2eb4ecbbe9a330f092(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ec2f4ccfb3412b2eb4ecbbe9a330f092(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ec2f4ccfb3412b2eb4ecbbe9a330f092(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ec2f4ccfb3412b2eb4ecbbe9a330f092(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a638c2be267e3a216631901b32676cb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1524, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1524, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a638c2be267e3a216631901b32676cb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1524, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1524, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ec2f4ccfb3412b2eb4ecbbe9a330f092(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dcbdef923e90a35e930c839bfa4f32fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 16, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a57b0676523ed4f4b4c56b57fb4a8364(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_182e5be9a78d93d86dc49cf8d001202a
    def get_inputs(self):
        return [
            paddle.to_tensor([2340], dtype='int32').reshape([1]),
            paddle.to_tensor([4], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_714cafee109e28966012a0677beb26de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_714cafee109e28966012a0677beb26de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_714cafee109e28966012a0677beb26de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_714cafee109e28966012a0677beb26de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_714cafee109e28966012a0677beb26de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b4db16db8ece64346376eef97b5c457d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2340, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2340, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b4db16db8ece64346376eef97b5c457d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2340, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2340, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_714cafee109e28966012a0677beb26de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_28e70d6740dc6404c75ba5563a0d8aca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_fd01f129d67cd9726390fea50756507a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 17, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b6ceef544aef363bbbccc049556e5093(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c318b9d63ec0327fe909f36e724ccf12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a6f2f992203e93b76e9c109a384b0c08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1024, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_41ea6b24a3670ae34cd395dc9d4ac27a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1024, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_747428e95f866c09dde566293ad73635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6301adf9aaada8f213617e4c888cae4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([6, 2304, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6, 2304, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d53cb3e2a4fbc20e99b1d46be09bf9c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([6, 2304, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6, 2304, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_95ff6127495f29d2b8f787d269e53d29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([6, 2304, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b613e6104520f4bc0aa6c15116ea99f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([171, 480, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([171, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0decf1ab0797f1ebd880929ceedaa96e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8571c6cb7aa88fe282f833ef5fb0d19c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 36, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b892123857428b0dff8a3bde2a723724(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d47107b8302089108238929c576ed55e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c72cb575f3c1974c9459dab9e0bfdfa7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_97c6575b53a9ff6fc1ea3ea92e99edbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f35d671a28baa87c850beee1b6b179de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 120, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f9766251d6d4f227cd5bc1acde9ea521(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b0b4f456bf0dca5bd1cabfe4acd7e872(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ad1b91ea104d7c603cbac75200700d74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9b8b0a4703e8fa516b70b48921cd8dec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7f55dd5cd3383c90beb035fd18cb71e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_691aabc5536874dca095d7f1d739eebf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5644e648d02e8e1db3eff74a0c0c84e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2bd78e6638ff7857b40e25a367bd82e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 5], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3314381f30756a64ba0ecc482064cd1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ee739dee6e2e4063dac5ae39ee6e0f08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_79b0dad505946dbf3024dd638287768f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 88, 132], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.09305113554000854], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_815ca8b26f00f9f06377fde58fe55dbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 19, 29], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.13885724544525146], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2db0b01467e59598ef5eeb5cedf14ded(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2db0b01467e59598ef5eeb5cedf14ded(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2db0b01467e59598ef5eeb5cedf14ded(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2db0b01467e59598ef5eeb5cedf14ded(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eddb038dad90ad66ccf9a4fcc2a44fd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_950041647603e17978e8b8728da1702f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 44, 44], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e98525176ba84e9f785ed5cc2a7987ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([145, 60, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([145, 60, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_51c7680adc50b5a9ac7dc9ec9cd55bad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([54, 197, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([54, 197, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c3b2fa4e1f84ad30a37b89126e1ee337(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([54, 197, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([54, 197, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_da447476dd298f672bad3fb2ea8387e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([54, 197, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_570d1038ca772afff599a3548bb618f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0b80d57590ad41e7f890d3f604280db2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 784, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1e67694fcf2d31455cc375b606a87135(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_62e44e7bc076d61ddcba63928f2f9788(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1605900228023529], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_536c61fa961b938ffc18c8215d1be8c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.5567916631698608], [0.2501413822174072], [-0.7649469375610352], [-0.20857638120651245]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.5128167867660522], [-0.0029732584953308105], [-0.1997864544391632], [-0.5080423355102539]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_9cc92902129ed21da8946ef7c802691e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1605900228023529], [-0.035677552223205566], [-0.5607033967971802], [0.7076130509376526]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.4928453266620636], [0.10515210032463074], [-0.21348506212234497], [-0.055697500705718994]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_a3db881d730dace64dbad8c969fa28ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.5567916631698608], [0.5858219861984253], [-0.5281831622123718], [0.7076130509376526]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.5128167867660522], [0.6068089008331299], [-0.11631205677986145], [0.17116692662239075]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_4e924fc884281c56ceb69b9945bf3ce2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 80, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d7352dd94434211fb041a08a13babfb1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7beb532d424b73dcc832e185454879b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([8816], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([8816], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6317290caaae01314e6e0806fe5d1cc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_95db775d4c3d310ab17a03afd448a558(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f509dbd7e0e0c9a27a06abcf61ad05a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9eea32ca1454f6528000f99c00a90664(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_97c6575b53a9ff6fc1ea3ea92e99edbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4a3bfff7d325b2313be34b7ad6ed2aef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 52, 52], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7e25941e61ec5448f6723dad3e75e174(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_48da4e0e198a70ac55704f1c80ddd5a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6867c07f499cbb344d0274143c66f195(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_97c6575b53a9ff6fc1ea3ea92e99edbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c35bbe9e62243c7a45589f8a18d6f185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 88, 88], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5b096df558a3fccb34a4d82aa845af3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.3373871445655823, -0.9848598837852478, -0.5651422739028931, -0.12044256925582886, -0.28104254603385925, -0.546388566493988], dtype='float32').reshape([6]),
            paddle.to_tensor([0.15108215808868408, 0.17850139737129211, -0.5844396352767944, 0.455766499042511, -0.590448796749115, -0.025779783725738525], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_0109f44db54c1330f6b60da3fee10cd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.05097317695617676, -0.17579886317253113, 0.3302915394306183, -0.05489368736743927, 0.1659412384033203, 0.014085778966546059], dtype='float32').reshape([6]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_db86771e7e5ff03ded56d42699d5d2ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.0, -0.0, 0.0, -0.0, 0.0, 0.0], dtype='float32').reshape([6]),
            paddle.to_tensor([1.0, 1.0, 0.0, 1.0, 0.0, 0.0], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_a954bc69dcefbe07371e317d7b7ecb0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.27928078174591064, 0.0, 0.0, 0.0, 0.18402773141860962, 0.0], dtype='float32').reshape([6]),
            paddle.to_tensor([0.7156945466995239, 0.17850139737129211, 0.0, 0.5968397855758667, 0.08256739377975464, 0.1728116273880005], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_53ebd671fc8b80473310cd23d358c64c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.3373871445655823, -0.9848598837852478, -0.1431266963481903, 0.1323724389076233, -0.13841024041175842, -0.2427496612071991], dtype='float32').reshape([6]),
            paddle.to_tensor([0.15108215808868408, 0.5066627264022827, -0.5844396352767944, 0.5531957149505615, -0.590448796749115, 0.5862129926681519], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_0c10b324619723ce6ccb641a339153ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.20891931653022766, 0.1278628706932068, 0.46038001775741577, -0.3298269808292389, 0.30385130643844604, 0.3849843740463257], dtype='float32').reshape([6]),
            paddle.to_tensor([0.20891931653022766, 0.1278628706932068, 0.46038001775741577, -0.3298269808292389, 0.30385130643844604, 0.3849843740463257], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_5869b64d9b584f24575cceb5c23a276c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.07249376177787781, 0.1131812334060669, -0.3551810085773468, -0.11925125122070312, -0.31808990240097046, 0.4052920937538147], dtype='float32').reshape([6]),
            paddle.to_tensor([0.07249376177787781, 0.1131812334060669, -0.3551810085773468, -0.11925125122070312, -0.31808990240097046, 0.4052920937538147], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_6a048e4b79c6bfeaeeeb8e2bc4a78723(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.27928078174591064, 0.0, 0.42201554775238037, 0.25281500816345215, 0.32666003704071045, 0.30363890528678894], dtype='float32').reshape([6]),
            paddle.to_tensor([0.27928078174591064, 0.0, 0.42201554775238037, 0.25281500816345215, 0.32666003704071045, 0.30363890528678894], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_46c1cf48bec9c6408739d16916cda15c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.7156945466995239, 0.5066627264022827, 0.0, 0.6942690014839172, 0.08256739377975464, 0.7848044037818909], dtype='float32').reshape([6]),
            paddle.to_tensor([0.7156945466995239, 0.5066627264022827, 0.0, 0.6942690014839172, 0.08256739377975464, 0.7848044037818909], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_df632dff66d1a4ebc698a1bbe53c2159(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.6167720556259155, -0.30232495069503784, 0.039032455533742905, 0.27651479840278625, -0.37237289547920227, 0.016711939126253128], dtype='float32').reshape([6]),
            paddle.to_tensor([-1.5218230485916138, -0.7459564208984375, 0.09630866348743439, 0.6822724342346191, -0.9187926650047302, 0.04123502969741821], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_19ad465f1ca87c14a6aff462a97282df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4841686189174652, 0.18402066826820374, 0.0037450853269547224, 0.15871542692184448, 0.25491762161254883, 0.0006886427290737629], dtype='float32').reshape([6]),
            paddle.to_tensor([0.9386179447174072, 0.2255212366580963, 0.0037591636646538973, 0.1886584311723709, 0.342133492231369, 0.0006891172961331904], dtype='float32').reshape([6]),
        ]


class PrimitiveOp_26c496af661f594a452b0002b6269924(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_960d9ee4b6e661d736fc06f1436d0e20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26c496af661f594a452b0002b6269924
    def get_inputs(self):
        return [
            paddle.uniform([22, 4, 1, 49, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 4, 16, 49, 56, 56], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_efc2c158b1006f1fa94d51c5dad3c992(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1ef29201c8d73c4f913d77e2232373e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 9, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_28dc3c38afa72020b6d5db75cc2a2e16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7b3466fd9569e8a6835c3db3c8144aa1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.41462886333465576], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ae3991a0a4dfd59ab2d0279df158facc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f13daf3f449b691a958a3282c3b2a4e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0d7fcd47d9d189d32bee6860fc356e8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 68, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7546d0b98fdf07b0d5cf61c6c88ef879(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8dbc9c6e93698784e2dc76cd90629c75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_22add1c61857bba0bc1a9da68aca4989(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26c496af661f594a452b0002b6269924
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 1, 49, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d13c7780040c7d5390d16013ff0096f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0a6391b635813f8fa87769d0b770d85a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26c496af661f594a452b0002b6269924
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 1, 49, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b33ca3ed7bd30b54dd2402666271ae9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([15200], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b38c54914b248cf4219c062f264f281b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ebf5432c2d577ed9dbf0b9d29ad39c7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 48, 48], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5a2de620b6d5b8dfa2bd8fe89555a496(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_bf702ec4ead2b73ce293cb39b16218f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c5ae01231dc6d96bd61e5fce16208ec3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_091b43d06d68700266baeb6fb91cb09d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 168, 168], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e1f714053a4f0006ee2e9ceee4b4036f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_098f6fb2b515e65295fb81d132593520(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_28e88801addf240b242bfa6ecec33815(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_efc2c158b1006f1fa94d51c5dad3c992(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_731aadfec617123f438328f1efe9071d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 22, 22], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_19c41d683a62a76fb9d46e50734ff96f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 480, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_28e70d6740dc6404c75ba5563a0d8aca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ae67d1fbbdc5985e8f3892a2c0204889(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c4867d17d343262fc1371c8bbaa146dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f5dc52f6f389fd7847349c734c5e82ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_52bb0faa52d939e4ffa93df7bd4a4807(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a89695a805036c4b9147972a750fc47f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_656c4b3a818a83e363f5be7d6cff2815(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bdf21a8ee7a6441046a162878f77082a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4981ac87ab41ae22d3852d490d340ca9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 15, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_20b94f481f85325a992c995860dcd103(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 640, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f50c25b50657c7a06c1cc63b29bde18c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 640, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d58e2b1d500e85cf5b3c9318066ca7eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0e66f5bd2c5458c016b3f9caef8ac217(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_94e08cb3e355367e3b95ea533ca5cac9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 9, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4854abcb4dd82b3eb081e6c1aa6ef9a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26c496af661f594a452b0002b6269924
    def get_inputs(self):
        return [
            paddle.uniform([22, 2, 1, 9, 112, 112], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 2, 16, 9, 112, 112], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ae3991a0a4dfd59ab2d0279df158facc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_97c6575b53a9ff6fc1ea3ea92e99edbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2735dc8929bd843f87d075bf84b749a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 104, 104], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4e6d1e96f848ce8a1c211ee9dedbaabb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 20, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.15902146697044373], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e1f714053a4f0006ee2e9ceee4b4036f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_098f6fb2b515e65295fb81d132593520(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_28e88801addf240b242bfa6ecec33815(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_58a0463bd3f145bb3c024a5081032881(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([145, 336, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([145, 336, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_49317fa304dc4897b66bc41120d87aec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_49317fa304dc4897b66bc41120d87aec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_49317fa304dc4897b66bc41120d87aec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_49317fa304dc4897b66bc41120d87aec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a6ae88e3b67c7699409325b3113ccaf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 13, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.2606169581413269], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_28dc3c38afa72020b6d5db75cc2a2e16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_34d2ed6168734195f79b4e4af4f6ffc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_34d2ed6168734195f79b4e4af4f6ffc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_34d2ed6168734195f79b4e4af4f6ffc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_34d2ed6168734195f79b4e4af4f6ffc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_34d2ed6168734195f79b4e4af4f6ffc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_07e622a39f9ba29f914053559426dc46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_87e766972942094b76aad348fa65744c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 13, 13], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1cc5b17da916be86eb5e754e164f3e11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([300, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1cc5b17da916be86eb5e754e164f3e11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([300, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b256de4a2000789eb540fed5e2c4bc9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([22, 240, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6a1dc02ac9aa61dc3f6c050e283c2491(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8e4e0d7fe4b352e5f6de35c7f56148bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2224c6ab901e9c5adb2c1c0f7f9d7f42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_472a7b87e9c4bd9112b14b028b27ffc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2547431bfdc47284d3fb9c65218eed9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 52, 52], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_56c63864e99dd90ca85a022076d58b7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 10, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.08099251985549927], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3f4f8eccba6b9722be4d0272edd45d41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1b76e73f5d5e8491910351c3f32685eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0decf1ab0797f1ebd880929ceedaa96e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5a5537e1868732b5f6e6189952505a11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 88, 88], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_392316234e12bd5e48b839f57d215a50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_392316234e12bd5e48b839f57d215a50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_392316234e12bd5e48b839f57d215a50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_392316234e12bd5e48b839f57d215a50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b892123857428b0dff8a3bde2a723724(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d47107b8302089108238929c576ed55e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c72cb575f3c1974c9459dab9e0bfdfa7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_16c93c8c8e373129e4dd235f015051c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 200, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_398c52a23ccb10cac96e8371a0ac6d2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 200, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eacbd32fb9ddfc4a554df0aaa162cf40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f38fe8b8d8bf66fbce9771ebeb676c86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9a0ca03fdde489f1562762f8e4810940(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_945ae0a273d54370b1c0e14cc73c0402(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3314381f30756a64ba0ecc482064cd1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_203c061b09b530ed269dafa6282154c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3540749566bc4995d48fc1381c7fd005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.13322335481643677], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_62e9e8175f43f450b55310247f16d6f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.19260317087173462], [-0.3981360197067261], [0.1206163763999939], [0.747077465057373], [-0.45611828565597534], [-0.4333042800426483]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.30148178339004517], [0.12030655145645142], [0.13322335481643677], [-0.43120890855789185], [0.09307345747947693], [0.003935098648071289]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_0259162c9e94309d321c2caf8f33ed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.13190007209777832], [0.0221560001373291], [-0.4627215564250946], [-0.05427950620651245], [0.3279499113559723], [0.056355416774749756]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.09038162231445312], [0.14011234045028687], [0.827140212059021], [-0.27585792541503906], [-0.1880994439125061], [-0.04197511076927185]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_8ce0b8a2d1f6116fb4cb96fa5a64d5db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.19260317087173462], [0.0221560001373291], [0.1206163763999939], [0.747077465057373], [0.3279499113559723], [0.1086500883102417]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3456398844718933], [0.551206111907959], [0.827140212059021], [0.04979658126831055], [0.5788179636001587], [0.12194603681564331]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_5f5d1ee7a72b78cbc78ef6efd8749dda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f824a5610ff48b3fafd9ec923f0bbc2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c4867d17d343262fc1371c8bbaa146dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_472a7b87e9c4bd9112b14b028b27ffc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_81c828554ad69b9b4b0efc4931a4e78a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6a1dc02ac9aa61dc3f6c050e283c2491(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0ddf20395e5ea8f242046ab79bc43a12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 20, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_97c6575b53a9ff6fc1ea3ea92e99edbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f36b9650bab12e04230bb5cf42d5e01e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 36, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_888801390788289605d10b7cd81591c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b679515687f5baa02eb7a6671afa090c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 20, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_90152b3eddce526cf3a2768f7b2f04f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5fb3ecfac004bcd348b0ec418a251f9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bb65d1edca49c2f2c6c4f9012a5b297b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eac3b86c36b90a240e34e4dfd998cc35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eac3b86c36b90a240e34e4dfd998cc35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eac3b86c36b90a240e34e4dfd998cc35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_14481e53021a6e33a3334c2484d4e83d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 112, 112], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_203c061b09b530ed269dafa6282154c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_28e70d6740dc6404c75ba5563a0d8aca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3a9ee5e4fb285299d7aa1ae30c59012c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 20, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ccef2fb8604b4dd22a7968f9d5c2f268(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e1e63d13ad47aa6f896ba2614299e53b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 48, 48], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_472a7b87e9c4bd9112b14b028b27ffc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c8b594f6a52c2af99d11de4ec413f589(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 40, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a5c4eb0e074163dbb338ea9e40ac7a15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([2.100213050842285, 1.7875185012817383, 1.9936890602111816, 1.9423627853393555], dtype='float32').reshape([4]),
            paddle.to_tensor([0.585746705532074, 0.5934618711471558, 0.6064386367797852, 0.6031104922294617], dtype='float32').reshape([4]),
        ]


class TestPrimitiveOp_8e11d18abd2273609a02d9d10d89f414(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([2.6189918518066406, 2.4944522380828857, 2.0712757110595703, 1.7656850814819336], dtype='float32').reshape([4]),
            paddle.to_tensor([0.414253294467926, 0.40653812885284424, 0.39356136322021484, 0.39688950777053833], dtype='float32').reshape([4]),
        ]


class TestPrimitiveOp_9914dd1ea13509355f7fba6639e0543b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5787796974182129, 0.518728494644165, 0.5060560703277588, 0.46806031465530396], dtype='float32').reshape([4]),
            paddle.to_tensor([-0.49916917085647583, -0.034904420375823975, -0.38272324204444885, -0.24604645371437073], dtype='float32').reshape([4]),
        ]


class TestPrimitiveOp_3314381f30756a64ba0ecc482064cd1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0ec1d5484fbf58f7fc7acbd4fcba0762(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_00b9e74919fdf6f4a27ff636669b12aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d3ae46feafa36ada9f7382c91ec1170d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 48, 48], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2c8d54494d8ea71dca0bb85b1293e005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 21824, 15], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2c8d54494d8ea71dca0bb85b1293e005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 21824, 15], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_54a6576ad3b51eebfa1a3b55454b681a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ec45491a69b3e908e66d7efbaa3daf5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54a6576ad3b51eebfa1a3b55454b681a
    def get_inputs(self):
        return [
            paddle.to_tensor(-1.0, dtype='float32').reshape([]),
            paddle.uniform([1, 21824, 15], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2c8d54494d8ea71dca0bb85b1293e005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 21824, 15], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7c2355798bf7d49cc5c80f27bffb28e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_04c68db8686049d6a4e7705c9f9a8d33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f31cde4239d56e97e1f617f2afde02c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5f5d1ee7a72b78cbc78ef6efd8749dda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_576249bc2f8b4d27cad5840fbc1d960b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 11, 11], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8e7d70d0d67ade69fea2a5791f0130d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 40, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e629737c1904b65c5e948b6ec0e4c9af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([22, 336, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 336, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bdf21a8ee7a6441046a162878f77082a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3b581385bc92e4cf01e0209abbf95b88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3314381f30756a64ba0ecc482064cd1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a205d02a855c4194e30cc352a2eed454(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6d3bc3d8306a2c0e4e79f6ad71d896ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([10, 60, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 60, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6cdcd041f4db9af3da74c304d57156fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 60800, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 60800, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1c69e1a1ddaa46098939022a18c342ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 60800, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 60800, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fd8e616776f0530ed03863e411710432(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([1, 60800, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_da6bd0ec51f68274ddb596c6a9897fe5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_460a5abce654dcb0e0729d2bf2d56262(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e13f93163f149910766dd5ac5f125925(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f9766251d6d4f227cd5bc1acde9ea521(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b0b4f456bf0dca5bd1cabfe4acd7e872(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ad1b91ea104d7c603cbac75200700d74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a7cbf9ea34c096a043368dc76a2781b7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c554a4d77e58a1d9411819ca01cb87bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7cbf9ea34c096a043368dc76a2781b7
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.27303221821784973], dtype='float32').reshape([1]),
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9b8b0a4703e8fa516b70b48921cd8dec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2832049614fd4273c0901f1886a24b5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3bc5c2142495ff854c0faec3cc62fe8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 784, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a5f42fbf3f89d2a7890d8fab32858fc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_88343f655d1a7faa9a2cb1ccfb643065(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 21504, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_88343f655d1a7faa9a2cb1ccfb643065(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 21504, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_be086fec8f9324007c0058934c2a5968(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_20c107a070b99279e4832b3b905cb66e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_10e9ad1f3e16c64500358a2cf873c22d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_13785115db9a95a2420c0e44c2676c1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_49c99ea77245c1f22640e0b4d063ef6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.10843027383089066]], [[0.5667374730110168]], [[0.4583495259284973]], [[-0.3406211733818054]], [[-0.08109785616397858]], [[-0.22424793243408203]], [[-0.13900572061538696]], [[-0.6688389778137207]], [[0.1526089608669281]], [[-0.18299663066864014]], [[0.4619065225124359]], [[0.4545930027961731]], [[0.3897501230239868]], [[-0.5024852752685547]], [[-0.5564124584197998]], [[0.5752174854278564]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ddaffa2e8fb7a1da73bb62e3f2116851(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 80, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[0.47831395268440247]], [[0.6133474707603455]], [[0.5916699171066284]], [[0.4318757653236389]], [[0.4837804436683655]], [[0.45515042543411255]], [[0.47219884395599365]], [[0.3662322163581848]], [[0.530521810054779]], [[0.463400661945343]], [[0.5923812985420227]], [[0.5909186005592346]], [[0.5779500007629395]], [[0.3995029330253601]], [[0.38871750235557556]], [[0.6150435209274292]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_b159e32d44acd4630dca3b0c597e5bdf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6, 9216, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1f81e9d9bddaa216a0332f95f619e8ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6, 9216, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_490b57f2ace88f10802290a52fc30bab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eac3b86c36b90a240e34e4dfd998cc35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eac3b86c36b90a240e34e4dfd998cc35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eac3b86c36b90a240e34e4dfd998cc35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2458f135fdff46b9500bf4750d1a4dce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_df684325b1d521738882e56d0283b61c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 21, 21], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e445294178da4c170e0ff6f646e1ad59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 112, 112], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8ef55ef2c77b3632c20fdcdfa107a7a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 1152, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bdf21a8ee7a6441046a162878f77082a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_fb4e37e79b4854655bbcdc728f5ee73f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 46, 46], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8ab165aae29fcb9f1fbed75a388a3408(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([551], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([551], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0cf2e61a0c9466b6bd1b728deb8c012f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.18415933847427368], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_90152b3eddce526cf3a2768f7b2f04f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5fb3ecfac004bcd348b0ec418a251f9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bb65d1edca49c2f2c6c4f9012a5b297b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_087508af6390cc5face5aa89a27e842f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[0.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_52bb0faa52d939e4ffa93df7bd4a4807(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a89695a805036c4b9147972a750fc47f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_656c4b3a818a83e363f5be7d6cff2815(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_97c6575b53a9ff6fc1ea3ea92e99edbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_693c8dd565a054da46ce216cb02fe2ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 256, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f9766251d6d4f227cd5bc1acde9ea521(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b0b4f456bf0dca5bd1cabfe4acd7e872(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ad1b91ea104d7c603cbac75200700d74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_888801390788289605d10b7cd81591c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_75a2a1e6870300247a349120d6418d49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b73d33b20043d292f386fb153fe5e420(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_bf91e5d2237581fb4d1bc33f6a3c7c2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 88, 88], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eddb038dad90ad66ccf9a4fcc2a44fd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_51ad6c48f6881178c2cbb9a20683dd1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1f450f6efe3f545afd284976548f6975(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.2163582444190979], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1f450f6efe3f545afd284976548f6975(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.2163582444190979], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_921ad6c9d6e2621ed08195154ac57b8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_522cac248c30b086690f7f50ffba2668(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.11153560876846313], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b410fb189fc466921833d2a9337e683d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 24, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3f4f8eccba6b9722be4d0272edd45d41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c188cbf0daa709b9bb21a15a22b415b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7c4306edff1fd7be57773bd644e60dfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2204, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2204, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7c4306edff1fd7be57773bd644e60dfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2204, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2204, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7c4306edff1fd7be57773bd644e60dfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2204, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2204, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dae065cad8db7b8e98213cee845b875a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[-0.45192790031433105, 0.14313971996307373, 0.4611530303955078, 0.2606866955757141]]], dtype='float32').reshape([1, 1, 4]),
        ]


class TestPrimitiveOp_472a7b87e9c4bd9112b14b028b27ffc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6fb74b062b9dbf17531d0b77028f3612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 80, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c8e1cc4ae6693cd0a02be72cff0a9878(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c8e1cc4ae6693cd0a02be72cff0a9878(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c8e1cc4ae6693cd0a02be72cff0a9878(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c8e1cc4ae6693cd0a02be72cff0a9878(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8e0113f39404be7344c1686ccd5835f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_52bb0faa52d939e4ffa93df7bd4a4807(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a89695a805036c4b9147972a750fc47f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_656c4b3a818a83e363f5be7d6cff2815(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bdf21a8ee7a6441046a162878f77082a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5904999ecf06989e0ce442c6d0ae9cad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 13, 13], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ef9025d6249cc539bd6dd1de707c0419(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 160, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a9335357df1b2dab018f7c278c041dc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 160, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ba89b3d64e03e6d3f1b2a9d9568438bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_472a7b87e9c4bd9112b14b028b27ffc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c38075391bbf59e28d41db7152965493(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 8, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7ce62a4c4e9cc08e2373e45b0a2b29c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 112, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b6ceef544aef363bbbccc049556e5093(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8e0113f39404be7344c1686ccd5835f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f714686ff184111aea28ca2a2f214edd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 16, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1722b6e7bf8580670a7a55d5851fc622(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([1.1404352188110352], dtype='float32').reshape([1]),
            paddle.to_tensor([-0.28722038865089417], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_97f6416c3311a2e10eac9b5fcb62f5b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([1.339958906173706], dtype='float32').reshape([1]),
            paddle.to_tensor([-0.3079986870288849], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ee38d026f60b7984ffd560b28b9b4d1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ee38d026f60b7984ffd560b28b9b4d1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ee38d026f60b7984ffd560b28b9b4d1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ee38d026f60b7984ffd560b28b9b4d1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0f45da8c2278bc84c00d60a10bfe59d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 169, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1eeb9d3fc7509eec63c4e02c1a9d7a9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 169, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_28cf51959033ebf36c5f8fd43cc2f216(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1024], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3a53d9a4a32d142a36ad555a2f662926(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7981073b3a6337da98cd3131e4c3c74b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a53d9a4a32d142a36ad555a2f662926
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 2048], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.7071070075035095, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_46863b0e126f0f581b604902b30c0d96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a53d9a4a32d142a36ad555a2f662926
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 2048], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.5, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7645a9c5859efe102cbb3a54c56b064d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 2048], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 169, 2048], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0f45da8c2278bc84c00d60a10bfe59d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 169, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1eeb9d3fc7509eec63c4e02c1a9d7a9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 169, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_28cf51959033ebf36c5f8fd43cc2f216(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1024], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7546d0b98fdf07b0d5cf61c6c88ef879(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_75dd301de17b2052c52205b4728c0005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2b1685ac7c4b2b3ba08d2776a3c751ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_182e5be9a78d93d86dc49cf8d001202a
    def get_inputs(self):
        return [
            paddle.to_tensor([2047], dtype='int32').reshape([1]),
            paddle.to_tensor([4], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_0f44a1c3f5c851e36f7eb87d22e2fb32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0f44a1c3f5c851e36f7eb87d22e2fb32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0f44a1c3f5c851e36f7eb87d22e2fb32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0f44a1c3f5c851e36f7eb87d22e2fb32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0f44a1c3f5c851e36f7eb87d22e2fb32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c3e910cfcb5971c21613dbd5d21774ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2047, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2047, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c3e910cfcb5971c21613dbd5d21774ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2047, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2047, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0f44a1c3f5c851e36f7eb87d22e2fb32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fd5cdbfb1e073a0706218d774bac947f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fd5cdbfb1e073a0706218d774bac947f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fd5cdbfb1e073a0706218d774bac947f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fd5cdbfb1e073a0706218d774bac947f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fd5cdbfb1e073a0706218d774bac947f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9a4f6260f784c8f1cc37bd527cd8040d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 624, 20, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 624, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_28e70d6740dc6404c75ba5563a0d8aca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_104967985847876967ceb7c9cfe53ffc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 19, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3cf832d13fae26b08c6d7ae7528aa450(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 46, 70], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.1842973232269287], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ef6810171e12c271503a6159fcac0d5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7456cdb7d758a2d4b7c68c211d327996(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 44, 44], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_84f84473e202121af57c08c697468fe0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 80, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[1.0]]], [[[0.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_b6ceef544aef363bbbccc049556e5093(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_86f9f1803fc2fe57859abc321c6f43fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 23, 23], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ccefe6c71ebf4c16628ad6285fa95dcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a328e8fc90f312f6c0b501ed7d4ed772(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 44, 44], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9f5d38e5e94a37e841def9503fc3da38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_960b283dc54573fa7a395018f95af8f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([171, 60, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([171, 60, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_276a930fd9a327be9f1505caac860e16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_276a930fd9a327be9f1505caac860e16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b73d33b20043d292f386fb153fe5e420(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6464ed5dfd349c7f51161c89a955b0fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 36, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9f851a52dfb9e7f73bf2b1563a47580c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([10, 336, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 336, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0884721cbea88713d0248b06bbeadc33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_630a2221f04973bf4feee9d0610be22f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 32768, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6a5d420592592bdc6b89bb2dc97bde89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 32768, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ee332593e63a80bafe4bc49b49f5851a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_df17922f099f7cf3ef6b5eacb138c730(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2041b6946153d6a3fdc3584c6125113b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 320, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fddad8de81aa2b059aa3f7a23b3ffca1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a9fd878518b71fdf18f49af38a07dfae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a9fd878518b71fdf18f49af38a07dfae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a9fd878518b71fdf18f49af38a07dfae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a9fd878518b71fdf18f49af38a07dfae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5a2de620b6d5b8dfa2bd8fe89555a496(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8b860a74b0f792e02e73e4cad41a84cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a7887095f18d53a886be472ff10bd2d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 38, 58], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.36978569626808167], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8a1bc5a4f5132663bfc54af2c2a6421a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_807f87c577b1ffa62819fdad159c3308(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 32, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_472a7b87e9c4bd9112b14b028b27ffc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8b2729c946d92737255162f85c1e3075(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 96, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7e18206798a0c9c120a11e6ee84e0256(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([950, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([950, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7e18206798a0c9c120a11e6ee84e0256(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([950, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([950, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7e18206798a0c9c120a11e6ee84e0256(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([950, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([950, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8ef55ef2c77b3632c20fdcdfa107a7a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 1152, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_204aab022d5262a7ddfa568497777e5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 256, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bbce7c59d57c58cd1c295fd175558070(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26c496af661f594a452b0002b6269924
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 1, 49, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c277e2613c9f741f142e4e79729501b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([145, 36, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([145, 36, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8d568f54870b0fb1b40a9ec02a82d17e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8d568f54870b0fb1b40a9ec02a82d17e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8d568f54870b0fb1b40a9ec02a82d17e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8d568f54870b0fb1b40a9ec02a82d17e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8d568f54870b0fb1b40a9ec02a82d17e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ffa38f4c70097cd9521aee1217060a2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([10, 336, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 336, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_472a7b87e9c4bd9112b14b028b27ffc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9e22d32952dad9bf7444d8ffc30312d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 112, 112], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6d04e3b5eda107315979d885d975e62f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([1.9443645477294922, 1.7839425802230835, 2.6318883895874023, 2.1903347969055176, 2.4088072776794434, 2.4295654296875, 2.192326068878174, 2.3659110069274902, 2.4666571617126465, 2.172107696533203, 2.4629592895507812, 2.2720413208007812, 2.1945748329162598, 1.7010372877120972, 2.3898468017578125, 2.1940054893493652, 2.050135612487793, 2.2596428394317627, 2.1239683628082275, 2.624786376953125], dtype='float32').reshape([20]),
            paddle.to_tensor([1.453017234802246, 0.7974290251731873, 1.1712852716445923, 1.0704615116119385, 1.3312643766403198, 1.1228134632110596, 0.8940941691398621, 1.140920877456665, 1.4338600635528564, 0.7197654843330383, 0.7784366607666016, 1.4111160039901733, 1.0776876211166382, 0.5402962565422058, 0.7593830823898315, 0.879764974117279, 0.575200080871582, 1.368240237236023, 0.9464995265007019, 0.7761016488075256], dtype='float32').reshape([20]),
        ]


class TestPrimitiveOp_6f1fac705dc8ffd76fe1efb77c1e8383(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([2.5296783447265625, 2.5422403812408447, 1.8175106048583984, 1.895407795906067, 2.3886313438415527, 1.7767192125320435, 1.8078911304473877, 2.084463357925415, 2.105135917663574, 2.2278523445129395, 2.229203462600708, 2.313298463821411, 2.5338122844696045, 1.7475553750991821, 1.85045325756073, 1.971555471420288, 1.8982934951782227, 2.029557466506958, 1.6462875604629517, 2.2386739253997803], dtype='float32').reshape([20]),
            paddle.to_tensor([-0.45301729440689087, 0.20257097482681274, -0.1712852418422699, -0.07046157121658325, -0.3312643766403198, -0.1228134036064148, 0.10590583086013794, -0.14092087745666504, -0.43386006355285645, 0.28023451566696167, 0.22156333923339844, -0.4111160337924957, -0.07768765091896057, 0.4597037434577942, 0.24061691761016846, 0.12023502588272095, 0.42479991912841797, -0.36824023723602295, 0.053500473499298096, 0.22389835119247437], dtype='float32').reshape([20]),
        ]


class TestPrimitiveOp_12fb234e7f378e2411fe8e9d8b2fea0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.41980183124542236, 0.48438793420791626, 0.6928448677062988, 0.5527788996696472, 0.6038726568222046, 0.6274359226226807, 0.5379030704498291, 0.6013932228088379, 0.6558766961097717, 0.5469323396682739, 0.6027919054031372, 0.5637699365615845, 0.5420550107955933, 0.4306054711341858, 0.5650148987770081, 0.5418148040771484, 0.49640828371047974, 0.5860923528671265, 0.524603009223938, 0.6345841288566589], dtype='float32').reshape([20]),
            paddle.to_tensor([0.08450585603713989, -0.41024503111839294, 0.12457305192947388, -0.08356040716171265, -0.4872434437274933, 0.3690185546875, -0.0334722101688385, -0.3443579077720642, 0.4501416087150574, 0.20555460453033447, 0.45454758405685425, -0.40513208508491516, -0.48741498589515686, -0.3471773862838745, -0.28873416781425476, 0.1437639594078064, 0.31268078088760376, -0.06409019231796265, 0.46390068531036377, 0.19187211990356445], dtype='float32').reshape([20]),
        ]


class TestPrimitiveOp_203c061b09b530ed269dafa6282154c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ed4f6821165c8872f8b7b741f8d3bdd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 14, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.4936593770980835], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_21d33e63557cdedb0c9e03c8bccc7c16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.16268768906593323], [0.0], [0.0], [0.08316034078598022], [0.0]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_7457781edce00d803e8bc80f12b92d14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.16268768906593323], [-0.5031480193138123], [0.7967080473899841], [0.3249094784259796], [0.26126596331596375]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.365852415561676], [-0.5575640201568604], [0.12530755996704102], [-0.6186395883560181], [-0.9083819389343262]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_2f3fbbd9832f240fda485d0a88e0876b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.6373436450958252], [0.6108297109603882], [-0.6940600872039795], [0.09215694665908813], [-0.14561891555786133]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.48728251457214355], [-0.7872181534767151], [0.002398192882537842], [0.38675469160079956], [-0.010190069675445557]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_624201bc1ea9071217502bfd6e8fdb74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.6373436450958252], [0.6108297109603882], [0.7967080473899841], [0.3339060842990875], [0.6604487895965576]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.1894747018814087], [-0.4006441831588745], [0.9229835271835327], [0.38675469160079956], [-0.010190069675445557]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_037c8e3cb6cb69f0257e336ca502e2a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 65536, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1e6aff79f941622d6dd8918fbf3bdea8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 65536, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_37b07d8b3e157a54e2fd76d88703afb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0decf1ab0797f1ebd880929ceedaa96e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8130e2e5ffa7c601a91c190c17058809(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d1e47c1cdb92d099ceb5818657000b57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([15200, 81], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([15200, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d1e47c1cdb92d099ceb5818657000b57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([15200, 81], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([15200, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_41816642619940f30c0c4565f2c485b9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1589b1008ea6a8c8989b2e5e72b79fb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41816642619940f30c0c4565f2c485b9
    def get_inputs(self):
        return [
            paddle.to_tensor(-1.0, dtype='float32').reshape([]),
            paddle.uniform([15200, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d1e47c1cdb92d099ceb5818657000b57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([15200, 81], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([15200, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d1899fd93a20b6d3558396c6e05167e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_182e5be9a78d93d86dc49cf8d001202a
    def get_inputs(self):
        return [
            paddle.to_tensor([1813], dtype='int32').reshape([1]),
            paddle.to_tensor([4], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_9bb6e9dc5c02171077147450073b3855(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9bb6e9dc5c02171077147450073b3855(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9bb6e9dc5c02171077147450073b3855(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9bb6e9dc5c02171077147450073b3855(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9bb6e9dc5c02171077147450073b3855(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6033ddf1cb4a30e76da929cd9700a4f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1813, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1813, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6033ddf1cb4a30e76da929cd9700a4f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1813, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1813, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9bb6e9dc5c02171077147450073b3855(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3314381f30756a64ba0ecc482064cd1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_efaadc6c337c933594476c507a657b38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 34, 34], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e8f29167e8fa167f49631d8c294c21f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26c496af661f594a452b0002b6269924
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 1, 49, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6b561e40cde00d480ab3f377acb90929(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 1152, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 1152, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_79e430611f9ad66fa098e1c41fb05090(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3c171ab1ce9f0606dbdba8de72f3ce66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_071669aa2251f4b789877bf2cb5d3213(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fa1ad3ac145ebb05486ce6d44c1acdcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 11, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.03869009017944336], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4900860032e0d7d791b017fb18b5c0fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[0.5253017544746399]], [[0.5676259398460388]], [[0.5847143530845642]], [[0.5652614235877991]], [[0.42242440581321716]], [[0.44565147161483765]], [[0.6647840142250061]], [[0.6176483035087585]], [[0.509120762348175]], [[0.5108110308647156]], [[0.6216911673545837]], [[0.5196260809898376]], [[0.5152294039726257]], [[0.6985682845115662]], [[0.6070578098297119]], [[0.6936555504798889]], [[0.6219314932823181]], [[0.5733457803726196]], [[0.41928356885910034]], [[0.44175446033477783]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_da61655ab478751c3f9b2bd109fd35aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c23d3d9c27300a121f908ce899822b1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 32, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_931e71a8168fdf06af5c9b56c9daf121(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([2.3119239807128906, 2.375248908996582, 2.351820945739746, 1.9172375202178955, 1.793831467628479, 2.3923773765563965, 1.6805720329284668, 2.2762186527252197, 1.5915459394454956, 1.9527442455291748, 1.6075315475463867, 1.9440325498580933, 1.7371188402175903, 1.7622685432434082, 2.2982025146484375, 2.082996129989624], dtype='float32').reshape([16]),
            paddle.to_tensor([1.2985445261001587, 0.5642490386962891, 1.1309654712677002, 1.324817180633545, 0.8894685506820679, 1.3473598957061768, 0.6663962006568909, 1.3547391891479492, 1.4753153324127197, 1.4501453638076782, 0.5657334327697754, 1.170414924621582, 1.0569055080413818, 0.500735878944397, 1.199779987335205, 0.7593986392021179], dtype='float32').reshape([16]),
        ]


class TestPrimitiveOp_15d9eacc17ac0b49c014185e7dc76d3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([1.7655611038208008, 1.759546160697937, 1.9547991752624512, 2.152381420135498, 2.516341209411621, 2.0173590183258057, 2.52864408493042, 1.909085988998413, 2.226600170135498, 2.580552816390991, 1.8221396207809448, 1.7775472402572632, 2.333157539367676, 2.694002151489258, 2.0037953853607178, 2.435807943344116], dtype='float32').reshape([16]),
            paddle.to_tensor([-0.2985445261001587, 0.43575096130371094, -0.1309654712677002, -0.3248171806335449, 0.11053144931793213, -0.34735989570617676, 0.33360379934310913, -0.35473915934562683, -0.47531527280807495, -0.4501453936100006, 0.4342665672302246, -0.17041495442390442, -0.05690556764602661, 0.499264121055603, -0.1997799277305603, 0.24060136079788208], dtype='float32').reshape([16]),
        ]


class TestPrimitiveOp_496b6673675447f3d9090b886f9ebf21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.6187593936920166, 0.5267389416694641, 0.6009542346000671, 0.46021467447280884, 0.46842285990715027, 0.6306609511375427, 0.4908730387687683, 0.6016137599945068, 0.3224237859249115, 0.4175347685813904, 0.42518216371536255, 0.4931010603904724, 0.4258002042770386, 0.5568624138832092, 0.5892547965049744, 0.5419707894325256], dtype='float32').reshape([16]),
            paddle.to_tensor([0.03794509172439575, -0.14701583981513977, -0.4894035756587982, -0.3647373616695404, -0.34574568271636963, -0.40052130818367004, 0.36810386180877686, 0.45556044578552246, -0.26616615056991577, -0.19630658626556396, -0.28590157628059387, -0.1786017119884491, 0.2523499131202698, 0.0743212103843689, -0.27840012311935425, 0.4476921558380127], dtype='float32').reshape([16]),
        ]


class TestPrimitiveOp_98e34553828eddec0052d50ecea0d4ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([4875], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4875], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_98e34553828eddec0052d50ecea0d4ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([4875], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4875], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_19c41d683a62a76fb9d46e50734ff96f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 480, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_74b944a4bd3204f491836f1fbf8d35d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([4, 2304, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 2304, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9768e151ef0d033ea5c26f3f88f5c4fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([4, 2304, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 2304, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_077e77344037f77d3e140561487d89bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([4, 2304, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_28e70d6740dc6404c75ba5563a0d8aca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3d3aea16d9ee71e611da4c680a3b8f4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3cc2f1bced9a68e0574e2fc2bf6a9f00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3f4f8eccba6b9722be4d0272edd45d41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_855eea731a783a473f582de3bc251b70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 68, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ab3b59010e654172bc45a1366819d623(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_186a1957ef39feb9d6d2fc4bb8a14b90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 21760, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 21760, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a2934ca6e197e94dde567537a5d6a76e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 21760, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 21760, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cb4ec07c17a51d4f08bd1cffab130b59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([1, 21760, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1cf62000aab88f2283583dd2240834f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26c496af661f594a452b0002b6269924
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 1, 49, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_472a7b87e9c4bd9112b14b028b27ffc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3c855cfc4f3eda164f4b981c4d7caaee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 256, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ea9d0ed230461c5a52d1f249ef5b3b09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 44, 66], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.2954709529876709], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d55b2ae10fbae3f398f3a5df2c09588b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26c496af661f594a452b0002b6269924
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 1, 49, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3519c62b6696edbcf27e57797bfd07e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([54, 198, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([54, 198, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dfb91dffa90b85d163afddae01409a19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([54, 198, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([54, 198, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_38b99d67bf315abffd4125fa862008a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([54, 198, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bdf21a8ee7a6441046a162878f77082a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_677ce436d6616b9b8c61b1ed6907cc2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 19, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_60325cba8d4f67ee1c53e60fd7df3b26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_182e5be9a78d93d86dc49cf8d001202a
    def get_inputs(self):
        return [
            paddle.to_tensor([3061], dtype='int32').reshape([1]),
            paddle.to_tensor([4], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_65525ab959df11190ff869438146ea36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_65525ab959df11190ff869438146ea36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_65525ab959df11190ff869438146ea36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_65525ab959df11190ff869438146ea36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_65525ab959df11190ff869438146ea36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_54acd20a330ffe94103a87b7c2591ed8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3061, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3061, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_54acd20a330ffe94103a87b7c2591ed8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3061, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3061, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_65525ab959df11190ff869438146ea36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eddb038dad90ad66ccf9a4fcc2a44fd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_77fb7122d2b629e9a267eaf37d348f1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9f4a8ce8fd8f549608a59f0962ace51c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9f4a8ce8fd8f549608a59f0962ace51c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9f4a8ce8fd8f549608a59f0962ace51c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9f4a8ce8fd8f549608a59f0962ace51c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ff6a05301b16eb38332dcdedb14e83e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([171, 240, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([171, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6d99aae7d99fa34ebe817a6e2411f9dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1024, 512], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a1029d3a3c43b9f62d03055d679192fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1024, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0470f3e12cc360d98b86575177b01540(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b9fa18dafbd1970b116cd5718470458b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2832049614fd4273c0901f1886a24b5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3bc5c2142495ff854c0faec3cc62fe8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 784, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a5f42fbf3f89d2a7890d8fab32858fc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3314381f30756a64ba0ecc482064cd1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9d3fe177bcea93e9c53f2d66cbd40556(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7232d6c032f30091af12526b10838d72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.4961127042770386], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7232d6c032f30091af12526b10838d72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.4961127042770386], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_22823079e76bf3f216a09ba43a49d591(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_74dc79dd0909e352e315d8ff9483346f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.026489436626434326], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_906e993f74ad0812375da843d6535fb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([10, 240, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_63b984a7ada9e47e37140b8dba69b858(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_65e957e9aaf25d8a80216305a7c66f36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 52, 52], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f38fe8b8d8bf66fbce9771ebeb676c86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9a0ca03fdde489f1562762f8e4810940(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_945ae0a273d54370b1c0e14cc73c0402(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0edceb2c4b0c6d1b4496a8c97c95147b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([8816, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([8816, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0edceb2c4b0c6d1b4496a8c97c95147b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([8816, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([8816, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0edceb2c4b0c6d1b4496a8c97c95147b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([8816, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([8816, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8ca73db692d4967c544b6369071018b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 5, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.33650097250938416], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1cf62000aab88f2283583dd2240834f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26c496af661f594a452b0002b6269924
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 1, 49, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_472a7b87e9c4bd9112b14b028b27ffc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_fc12506dd4e25e99cb90b509ef56b545(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 68, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ae723d1f4e2a0af986110d50ff3a1e16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d66569920e5031c31a8c849401aa1471(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 100, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5f566e56ae9df223c4d0d17844ec0901(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8e03a38e48795fd7d76ac09c96d8030f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.01559913158416748], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8e03a38e48795fd7d76ac09c96d8030f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.01559913158416748], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_64da55d85736790a5a886937ad1abc9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_34308137311d5b1845b524f371edc18b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.09470519423484802], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f13daf3f449b691a958a3282c3b2a4e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1d359c42c942e1945568b8ffc536ca25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 40, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8b035cba6e0eaef3ef1c2ec8b9d885bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.07215607166290283], [0.0], [0.0], [0.019590437412261963], [0.0], [0.0], [0.0], [0.05962526798248291], [0.0]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.08150973916053772]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_89dd6de1429ebccb57a810f51cdcde88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3918963372707367], [0.2506285011768341], [-0.6258493661880493], [0.05717068910598755], [-0.2745057940483093], [-0.07387518882751465], [0.6177127361297607], [0.18446579575538635], [-0.1368616819381714]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.26183488965034485], [-0.26068878173828125], [-0.7445532083511353], [-0.03158220648765564], [-0.3763255476951599], [0.497799813747406], [0.6330482363700867], [0.1807483434677124], [0.15374690294265747]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_c5958f5dd47b40fb1890f5341aa47462(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2953508496284485], [-0.2269076704978943], [0.28040292859077454], [0.33574116230010986], [0.20336639881134033], [-0.037004947662353516], [-0.3291023373603821], [0.20334431529045105], [-0.683196485042572]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.3334689140319824], [-0.39746853709220886], [-0.1969008445739746], [-0.6603097915649414], [0.10665804147720337], [-0.6437492966651917], [-0.21342188119888306], [-0.5940254926681519], [0.43903985619544983]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_339f2e20d230a31b780e568a5b9b0640(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.61509108543396], [0.2506285011768341], [0.28040292859077454], [0.37332141399383545], [0.20336639881134033], [0.5067383050918579], [0.6177127361297607], [0.3281848430633545], [-0.1368616819381714]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.26183488965034485], [-0.056746602058410645], [-0.1969008445739746], [0.10791593790054321], [0.10665804147720337], [0.497799813747406], [0.6330482363700867], [0.1807483434677124], [0.5112770199775696]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_ef6810171e12c271503a6159fcac0d5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_96520cc9ad713adb64bb6194458e44fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 18, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b159e32d44acd4630dca3b0c597e5bdf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6, 9216, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1f81e9d9bddaa216a0332f95f619e8ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6, 9216, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_490b57f2ace88f10802290a52fc30bab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_63b984a7ada9e47e37140b8dba69b858(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c404fa57656737a59c21be42864e6607(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 60, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_96d34d59de4d68744c616e640c336307(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.3477025628089905], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_96d34d59de4d68744c616e640c336307(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.3477025628089905], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a2bb1bc986102c9fbe25b46b21268a8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_53c9585f26bd5acef628b765a005b15b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.16166889667510986], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_239ac5698c67a03d8dc66aa658afdf27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_97a4cad72cba275f5281ae5bb11c8dcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_877d979d9aba0c4aba2aa7af74202468(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_14481e53021a6e33a3334c2484d4e83d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 112, 112], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bdf21a8ee7a6441046a162878f77082a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c8a39c7144b7b44a49fffdce76ab4460(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b38c54914b248cf4219c062f264f281b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_fdebfdf5c30bfe6b1863cfe9bc218e86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_24fc7bb4d8ad73acfb0921118bde0314(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f13daf3f449b691a958a3282c3b2a4e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_da10084706a3360732c6e1ca1f4b226e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 76, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0a6391b635813f8fa87769d0b770d85a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26c496af661f594a452b0002b6269924
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 1, 49, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9d3fe177bcea93e9c53f2d66cbd40556(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_49f4e3acf382d504aaa18a1dca032071(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 50, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_65830a6968b1327e411ea6c7cf42738b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 50, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_930e5e428b9ad3a0dcc2a129d82aa5a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_28e70d6740dc6404c75ba5563a0d8aca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c4ef60302d8fba89287d11caed9f0baa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 11, 11], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_888801390788289605d10b7cd81591c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_35253fb0f73668ac5ccfb6e3d47ee271(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 34, 34], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5a2de620b6d5b8dfa2bd8fe89555a496(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d5be053475e125ce3611221a0ed97a22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4261c51004d092c451ec3407b052179a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 40, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_b73d33b20043d292f386fb153fe5e420(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c856a1f27dcd87e7eb43b4769166278b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_32fec282a9c8a43309fafb6922196583(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_570d1038ca772afff599a3548bb618f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0b80d57590ad41e7f890d3f604280db2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 784, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1e67694fcf2d31455cc375b606a87135(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aba80a094d2ef5af4d33df8d553cf980(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.11895418167114258], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e44718143120f147e1a39cfe77289194(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1f0ef64bc69006b95d45abd80b1b8b31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_182e5be9a78d93d86dc49cf8d001202a
    def get_inputs(self):
        return [
            paddle.to_tensor([2062], dtype='int32').reshape([1]),
            paddle.to_tensor([4], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_952c405ffe45b88994657acdc160662e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_952c405ffe45b88994657acdc160662e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_952c405ffe45b88994657acdc160662e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_952c405ffe45b88994657acdc160662e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_952c405ffe45b88994657acdc160662e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ed64f460839991e1f38ca421d4225b22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2062, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2062, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ed64f460839991e1f38ca421d4225b22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2062, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2062, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_952c405ffe45b88994657acdc160662e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bac4a44d6c5b3bc71de753802cfe9353(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 13, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.4477088451385498], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ccefe6c71ebf4c16628ad6285fa95dcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b3aa1a0a4abcb6ff59149e408de68c7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b73d33b20043d292f386fb153fe5e420(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3171972674f6eaceddd1ff0d8284ae57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 30, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a9fd878518b71fdf18f49af38a07dfae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a9fd878518b71fdf18f49af38a07dfae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a9fd878518b71fdf18f49af38a07dfae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a9fd878518b71fdf18f49af38a07dfae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a9fd878518b71fdf18f49af38a07dfae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1aa6928067144a49fc9e2ec5eac36f29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_93360fd56acab111dfce2ff4e164565f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_392316234e12bd5e48b839f57d215a50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_392316234e12bd5e48b839f57d215a50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_392316234e12bd5e48b839f57d215a50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_392316234e12bd5e48b839f57d215a50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_392316234e12bd5e48b839f57d215a50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5a2de620b6d5b8dfa2bd8fe89555a496(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9cee1ed6aa16a00bdad9f0f60fa8811c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 256, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_337c23fcb3c6108117d04e547be463ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_337c23fcb3c6108117d04e547be463ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_337c23fcb3c6108117d04e547be463ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_337c23fcb3c6108117d04e547be463ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0c4b3ee8fee909e4da72bbe2f6743ade(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 40, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.1680210828781128], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3314381f30756a64ba0ecc482064cd1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_860d0bd24c01614381bcc721c576b752(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 52, 52], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3be31b1998eda02a29996b4342be4b7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([150, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([150, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3be31b1998eda02a29996b4342be4b7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([150, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([150, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3be31b1998eda02a29996b4342be4b7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([150, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([150, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7f55dd5cd3383c90beb035fd18cb71e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_691aabc5536874dca095d7f1d739eebf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5644e648d02e8e1db3eff74a0c0c84e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dcf82fc16d535f93811662367f489572(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1174, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b9208762b373f437f6df56942271800e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1174, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6f9aafeef2eabbe1807be081cc5c024a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d55b2ae10fbae3f398f3a5df2c09588b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26c496af661f594a452b0002b6269924
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 1, 49, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f13daf3f449b691a958a3282c3b2a4e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_bcef493c8eafd3b0df91b91f3415dd15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 44, 44], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d3a93b5dee4c769b7ccbc0a6b32c2e23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 22, 33], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.2634822130203247], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c46d9a06d71efc1a927e941c859ba4dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 56, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.2599566578865051], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c9556eaac7526d658dc958cf66117103(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d88923f965106b39924a4d6b5fc1dd00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_407a9ff796d5cf2f8011d10f7742b6a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_407a9ff796d5cf2f8011d10f7742b6a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_407a9ff796d5cf2f8011d10f7742b6a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_407a9ff796d5cf2f8011d10f7742b6a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_800abd6197422f1c7554a644be93618b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([950], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([950], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_07e8d456b3256cfd2e3cefe422df5377(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([145, 480, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([145, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_472a7b87e9c4bd9112b14b028b27ffc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_fd6ae2ca77835f0b43757ddaddb69ef1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c5ae01231dc6d96bd61e5fce16208ec3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2801ec1b538d97547250c67dca4eaa42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b90085f987eae49f9ab3319dd15fa885(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c5da2438d2eaa210f217976100b832db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5f5d1ee7a72b78cbc78ef6efd8749dda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4f91c312f6e08872d2b23b9e8568f53b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_888801390788289605d10b7cd81591c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2c75bfda635d1f334886e135c902b3cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 38, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8a1bc5a4f5132663bfc54af2c2a6421a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eed5562801c9c8574c437ce90eb046f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([86, 198, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([86, 198, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1ffbce1890af2a6d22c98cc104af040e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([86, 198, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([86, 198, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_13d35a5623bf24d7ca1db98975e89906(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([86, 198, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b815cff9baec6ec5c4ff7899b6cb312e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([10, 36, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 36, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_03841210a13d8bef89ceb15e2ed15155(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ab3b59010e654172bc45a1366819d623(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_07e622a39f9ba29f914053559426dc46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ecfaadda65bbb552c7f24d8fa6430409(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 15, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bdf21a8ee7a6441046a162878f77082a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9809755eff56dc027593da6f3c97abf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_52bb0faa52d939e4ffa93df7bd4a4807(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a89695a805036c4b9147972a750fc47f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_656c4b3a818a83e363f5be7d6cff2815(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_41b682c3955cd21b29c8f21abe0434df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([247], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([247], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f5dc52f6f389fd7847349c734c5e82ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_888801390788289605d10b7cd81591c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_69ddfcbb672445670e53af60b8c2aaf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0e9a737a526bba489396a276bf64f07d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[0.45751017332077026]], [[0.449603796005249]], [[0.5518656373023987]], [[0.5184506773948669]], [[0.6168705821037292]], [[0.544891357421875]], [[0.5481473207473755]], [[0.4132453501224518]], [[0.4513944685459137]], [[0.4279438853263855]], [[0.4238579273223877]], [[0.3926095962524414]], [[0.38334745168685913]], [[0.5846882462501526]], [[0.5623013377189636]], [[0.5934792757034302]], [[0.4250046908855438]], [[0.48607295751571655]], [[0.5112768411636353]], [[0.5199869871139526]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_da61655ab478751c3f9b2bd109fd35aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c23d3d9c27300a121f908ce899822b1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 32, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f87a8eb0448d01d557410236d9f0129f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 16, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e218516152f0e97a3989469099de06ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([70, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([70, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e218516152f0e97a3989469099de06ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([70, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([70, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e218516152f0e97a3989469099de06ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([70, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([70, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d15804ea81f9f2e2ec7d50cd1331ebd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 9216, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a51395e0b89448221529248f3d87a068(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 9216, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_41656768658f72557eb1b186ae2b013d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_00f1183588fa2649d5af273f19741248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4313755edb98cac9fbeaa4bb73d2e0b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 30, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_00f1183588fa2649d5af273f19741248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f92d83adbfa213ede4d7185073c18d69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 60, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_00f1183588fa2649d5af273f19741248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c3df28b51639fe22df3304cf442ecaa8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 120, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_00f1183588fa2649d5af273f19741248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_247fb373629e5db681a3835ac0418a08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 240, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ab54b7500aa1bbf538f4900d593ce4e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_cc8510673aa57230431b5178e4552180(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 30, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ab54b7500aa1bbf538f4900d593ce4e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9dfd9dc17976e8e80b86ef27a7ed1569(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 60, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ab54b7500aa1bbf538f4900d593ce4e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0ee8fa772692252398b28e7d6376a114(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 120, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ab54b7500aa1bbf538f4900d593ce4e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b7c99be39798283c13f56a142504ca6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 240, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b90085f987eae49f9ab3319dd15fa885(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1d3af548529a45a502fbe8ba82931055(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 30, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b38c54914b248cf4219c062f264f281b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_64a81ef948acca1badee5558cfc349c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 60, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5f5d1ee7a72b78cbc78ef6efd8749dda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3a246bf41bd2c1c33ae3a2551fef4935(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 17, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bdf21a8ee7a6441046a162878f77082a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_eb555821a1a44ee39e7c2859461c96ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3314381f30756a64ba0ecc482064cd1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_194336844560b4b6974fa298f28956e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2e76701cfcae4f0944d92decff322fe2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 7, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.03636413812637329], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6b561e40cde00d480ab3f377acb90929(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 1152, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 1152, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bdf21a8ee7a6441046a162878f77082a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_64f5ac1a844e9be5c2883e231a975a32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e97fb7d17f0f99ff436c5acb568636a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_672d1e3bc7f3800a8119d10b23400b2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cc1f642995cc839146a7a5d59a207967(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cc1f642995cc839146a7a5d59a207967(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cc1f642995cc839146a7a5d59a207967(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cc1f642995cc839146a7a5d59a207967(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cc1f642995cc839146a7a5d59a207967(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a0e9a746cbcaeba191a40d493ac9615d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a0e9a746cbcaeba191a40d493ac9615d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a0e9a746cbcaeba191a40d493ac9615d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a0e9a746cbcaeba191a40d493ac9615d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a0e9a746cbcaeba191a40d493ac9615d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f6482ceaadbebd80080a973ef33e16a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7cbf9ea34c096a043368dc76a2781b7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3150287866592407], dtype='float32').reshape([1]),
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3f4f8eccba6b9722be4d0272edd45d41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d13853a44f299e6d5a8cb4faf97251ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 44, 44], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_28e70d6740dc6404c75ba5563a0d8aca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6fd8b9a9c34d1840126b853d6a9bdf9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0f7526d65c763ca9ec384b8a02ab6bf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 5], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2224c6ab901e9c5adb2c1c0f7f9d7f42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b073031ef0b6591773e92427bc7d210f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_97462771cbb3efb822221ff9eed0b027(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.37545904517173767]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.05390709638595581]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_fd1274b985bed985eabcdbf828ea584d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.308651328086853]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.7754269242286682]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_4ff25aae36f5a763d6579b8b2f9d1242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.308651328086853]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.05390709638595581]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_f8d330dc95d75a20fbd7594f4f7abcaf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.46796590089797974], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da6bd0ec51f68274ddb596c6a9897fe5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_460a5abce654dcb0e0729d2bf2d56262(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e13f93163f149910766dd5ac5f125925(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_20c107a070b99279e4832b3b905cb66e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_10e9ad1f3e16c64500358a2cf873c22d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_13785115db9a95a2420c0e44c2676c1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_48da4e0e198a70ac55704f1c80ddd5a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_48da4e0e198a70ac55704f1c80ddd5a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_48da4e0e198a70ac55704f1c80ddd5a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_529e62f93e3bc7cb29ae1bb29e30e553(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 2048, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_49f4e3acf382d504aaa18a1dca032071(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 50, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_65830a6968b1327e411ea6c7cf42738b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 50, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_930e5e428b9ad3a0dcc2a129d82aa5a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_28e70d6740dc6404c75ba5563a0d8aca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_bc0bffbe8ec2c7f93affdefde76fd2ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d2169de0d358e0bfde42924ba59f0bd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f13daf3f449b691a958a3282c3b2a4e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e0b02d67f2b485eb41af97168d10b661(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 20, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2f3d107057a208985201404208bcab52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a0e9a746cbcaeba191a40d493ac9615d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a0e9a746cbcaeba191a40d493ac9615d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a0e9a746cbcaeba191a40d493ac9615d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a0e9a746cbcaeba191a40d493ac9615d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7f55dd5cd3383c90beb035fd18cb71e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_691aabc5536874dca095d7f1d739eebf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5644e648d02e8e1db3eff74a0c0c84e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5a2de620b6d5b8dfa2bd8fe89555a496(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_fb0ed1e4de17662b26b1bbd109a31195(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 160, 160], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b658168f8862d312eaef001ed0c881cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([6, 576, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6, 576, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dca1ae21e5288241a8ae684ad03e33a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([6, 576, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6, 576, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6b8813c493d17892193ec08f34683ed6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([6, 576, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c9556eaac7526d658dc958cf66117103(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1a3ebe88b2b70ee9ab8cc4292788b7c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1025, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f69cdf7931b321c6db16162d69cac301(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1025, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3f24a59d133e72ee4047f50955a978d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e97fb7d17f0f99ff436c5acb568636a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e97fb7d17f0f99ff436c5acb568636a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e97fb7d17f0f99ff436c5acb568636a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_115a85defae7e8ed38baab5b3303e7f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 2048, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fd5cdbfb1e073a0706218d774bac947f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fd5cdbfb1e073a0706218d774bac947f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fd5cdbfb1e073a0706218d774bac947f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fd5cdbfb1e073a0706218d774bac947f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b90085f987eae49f9ab3319dd15fa885(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d2169de0d358e0bfde42924ba59f0bd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eddb038dad90ad66ccf9a4fcc2a44fd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7e2b5ad53f60495ea7a38faba7d0fcf6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 84, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cae69bcadf2689413baafc0da6876b53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([70], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([70], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bdf21a8ee7a6441046a162878f77082a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_35d61ed0bdb4a095a6c81f2b82e67ac3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8edaab6253c368457293c6409d832bf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 28, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.2595697045326233], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f13daf3f449b691a958a3282c3b2a4e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c88343644726655460c5dd22fc972bc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5c0e0fa8e7060cd7f2dbeeb52c0436a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5c0e0fa8e7060cd7f2dbeeb52c0436a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5c0e0fa8e7060cd7f2dbeeb52c0436a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5c0e0fa8e7060cd7f2dbeeb52c0436a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_30ab264989d34c1757ac90792da2b8b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_182e5be9a78d93d86dc49cf8d001202a
    def get_inputs(self):
        return [
            paddle.to_tensor([5526], dtype='int32').reshape([1]),
            paddle.to_tensor([4], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_64b7be1551a993febd2cb375642df1b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_64b7be1551a993febd2cb375642df1b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_64b7be1551a993febd2cb375642df1b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_64b7be1551a993febd2cb375642df1b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_64b7be1551a993febd2cb375642df1b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fa487beaa9be89a64be0a308e2219a01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([5526, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5526, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fa487beaa9be89a64be0a308e2219a01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([5526, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5526, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_64b7be1551a993febd2cb375642df1b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3f4f8eccba6b9722be4d0272edd45d41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e7bd8c7a76be6512ca19249708b74ebe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 76, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_03841210a13d8bef89ceb15e2ed15155(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_49c048802393c04f66ae71315ea851f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 42, 42], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ee38d026f60b7984ffd560b28b9b4d1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ee38d026f60b7984ffd560b28b9b4d1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ee38d026f60b7984ffd560b28b9b4d1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ee38d026f60b7984ffd560b28b9b4d1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ee38d026f60b7984ffd560b28b9b4d1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_256b90bf408e430a3920c2775b1b9b08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 5, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.26715087890625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_888801390788289605d10b7cd81591c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ee972681593e96b87f346ab816bb91af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2458f135fdff46b9500bf4750d1a4dce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_24fc7bb4d8ad73acfb0921118bde0314(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5fc2683eb41eba000fbdcd5cc25ff396(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 24, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_e4104d6050bdcc3cce154f4b6fa9eb81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 112, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_cc1f642995cc839146a7a5d59a207967(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cc1f642995cc839146a7a5d59a207967(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cc1f642995cc839146a7a5d59a207967(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cc1f642995cc839146a7a5d59a207967(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1f52d48f8e2597ab0f130b81d9bf8105(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_182e5be9a78d93d86dc49cf8d001202a
    def get_inputs(self):
        return [
            paddle.to_tensor([1071], dtype='int32').reshape([1]),
            paddle.to_tensor([4], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_a9d9c7390b37f7051406a131220abad0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a9d9c7390b37f7051406a131220abad0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a9d9c7390b37f7051406a131220abad0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a9d9c7390b37f7051406a131220abad0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a9d9c7390b37f7051406a131220abad0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_34fc01e08ef05abf2adfc986a24e0df6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1071, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1071, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_34fc01e08ef05abf2adfc986a24e0df6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1071, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1071, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a9d9c7390b37f7051406a131220abad0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9518e7ecea8f39cbaab241c1c775b45a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a3652ef1a2e693c36c3c280c9789fea2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 20, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ccef2fb8604b4dd22a7968f9d5c2f268(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_63d56d9d00157b2ec9784d81f346ea77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_32fec282a9c8a43309fafb6922196583(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_28e70d6740dc6404c75ba5563a0d8aca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4bb7c07a794dd0f131c84a07a58727d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 38, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_239ac5698c67a03d8dc66aa658afdf27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_97a4cad72cba275f5281ae5bb11c8dcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_877d979d9aba0c4aba2aa7af74202468(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e445294178da4c170e0ff6f646e1ad59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 112, 112], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_53a497af60fb10e5862b052745771c01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 97, 97], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c6703f7e20148c0eb28416cc9c5faba0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_182e5be9a78d93d86dc49cf8d001202a
    def get_inputs(self):
        return [
            paddle.to_tensor([1760], dtype='int32').reshape([1]),
            paddle.to_tensor([4], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6f60287ab05f203d89e619bc00c73b3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6f60287ab05f203d89e619bc00c73b3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6f60287ab05f203d89e619bc00c73b3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6f60287ab05f203d89e619bc00c73b3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6f60287ab05f203d89e619bc00c73b3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ceb7e4e3b40feb472f40d4cbe0e22d02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1760, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1760, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ceb7e4e3b40feb472f40d4cbe0e22d02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1760, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1760, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6f60287ab05f203d89e619bc00c73b3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_03841210a13d8bef89ceb15e2ed15155(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_828c6cb62cc9a9be79c5554c18eee7be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 22, 22], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4f111f1a4daeb2034f6a91c2af3cf813(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7a9bc5f7e6fd4c28949d895ea174f43b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([86, 197, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([86, 197, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2bc3fd82e9dc5790e33fd200cd6d4f1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([86, 197, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([86, 197, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1ec1ab1ec8f439d45f897e72ca5cc4d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([86, 197, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_be086fec8f9324007c0058934c2a5968(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_276a930fd9a327be9f1505caac860e16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_276a930fd9a327be9f1505caac860e16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0decf1ab0797f1ebd880929ceedaa96e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9ae6dfe35eee41bc7425399980bb38ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 52, 52], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cbb5bcc1f495bd5eaaad60809a520f00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bdf21a8ee7a6441046a162878f77082a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f3da89462997de38e91bd358ba70c191(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 17, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2db0b01467e59598ef5eeb5cedf14ded(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2db0b01467e59598ef5eeb5cedf14ded(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2db0b01467e59598ef5eeb5cedf14ded(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2db0b01467e59598ef5eeb5cedf14ded(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2db0b01467e59598ef5eeb5cedf14ded(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aeb1cff7a1df6229d8c116a6f4c0cf01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 112, 160], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.25803613662719727], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f38f39be8b5d4067f32e8fdde73897a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 10, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.23549973964691162], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b6ceef544aef363bbbccc049556e5093(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0ff0980efa3c353b685669e2eb6471dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 13, 13], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a528287ae057760d80d4954d9b0e8523(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([1.957185983657837, 1.9411578178405762, 2.114572048187256, 2.2039220333099365, 2.1572160720825195, 1.8176789283752441, 2.5826215744018555, 2.1649177074432373, 1.9631690979003906, 2.4696550369262695, 2.4230475425720215, 2.51082444190979, 2.059685230255127, 1.8122620582580566, 2.163853645324707, 2.388855457305908, 2.312471389770508, 2.1497910022735596, 2.4567131996154785, 2.223104476928711, 2.372714042663574, 2.4783079624176025, 1.8529603481292725, 2.5603344440460205], dtype='float32').reshape([24]),
            paddle.to_tensor([1.2453815937042236, 1.3311846256256104, 0.6801990866661072, 1.2272982597351074, 1.4603445529937744, 0.8681018948554993, 1.4291613101959229, 0.661310613155365, 0.5881569981575012, 0.5689818859100342, 0.8357322812080383, 0.805552065372467, 0.6140429377555847, 0.7574213743209839, 1.4061076641082764, 0.7503636479377747, 1.0906600952148438, 0.9193711280822754, 0.9379920959472656, 1.0703617334365845, 0.9862653613090515, 0.8883845806121826, 0.5237210392951965, 1.2894998788833618], dtype='float32').reshape([24]),
        ]


class TestPrimitiveOp_4f43434d02b4219007a862edc8f3d4a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([1.8278182744979858, 2.3267717361450195, 2.184431791305542, 2.185864210128784, 2.28001070022583, 2.6956605911254883, 2.5583858489990234, 1.8730792999267578, 1.9550988674163818, 1.9858431816101074, 1.8945577144622803, 2.084632158279419, 1.9350028038024902, 1.9535166025161743, 2.2005913257598877, 1.6777762174606323, 2.3635902404785156, 1.8516813516616821, 2.2743372917175293, 1.682910680770874, 1.6767299175262451, 1.7650961875915527, 2.270203113555908, 2.2520968914031982], dtype='float32').reshape([24]),
            paddle.to_tensor([-0.2453816533088684, -0.3311845660209656, 0.3198009133338928, -0.2272983193397522, -0.4603445529937744, 0.13189810514450073, -0.42916133999824524, 0.338689386844635, 0.4118430018424988, 0.4310181140899658, 0.16426771879196167, 0.19444793462753296, 0.3859570622444153, 0.2425786256790161, -0.40610766410827637, 0.24963635206222534, -0.09066009521484375, 0.08062887191772461, 0.062007904052734375, -0.07036173343658447, 0.013734638690948486, 0.11161541938781738, 0.47627896070480347, -0.2894998788833618], dtype='float32').reshape([24]),
        ]


class TestPrimitiveOp_faa8b76fd67d5cb5b126868bcc30ea5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4972326159477234, 0.4533621370792389, 0.5342283248901367, 0.5520066022872925, 0.5251721143722534, 0.4833707809448242, 0.6482555866241455, 0.516518771648407, 0.4899613559246063, 0.5652808547019958, 0.5840584635734558, 0.6069880127906799, 0.5028908252716064, 0.4616318345069885, 0.5372335314750671, 0.5528360605239868, 0.5769592523574829, 0.5314387083053589, 0.6113511323928833, 0.5652783513069153, 0.590788722038269, 0.5996756553649902, 0.5129210948944092, 0.662392258644104], dtype='float32').reshape([24]),
            paddle.to_tensor([-0.4691789448261261, -0.44313108921051025, -0.3143312335014343, 0.18662899732589722, 0.2776693105697632, -0.11237949132919312, -0.3546341061592102, -0.3077823519706726, 0.14455485343933105, -0.0022841691970825195, 0.06872892379760742, 0.008876562118530273, 0.14731431007385254, -0.48855656385421753, -0.32881948351860046, 0.21892493963241577, 0.379055917263031, -0.21522673964500427, 0.40095531940460205, 0.1536043882369995, 0.07284194231033325, 0.39020997285842896, -0.30889588594436646, 0.10605078935623169], dtype='float32').reshape([24]),
        ]


class TestPrimitiveOp_e8f29167e8fa167f49631d8c294c21f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26c496af661f594a452b0002b6269924
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 1, 49, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_77fb7122d2b629e9a267eaf37d348f1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f13daf3f449b691a958a3282c3b2a4e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_22f405542fa97f68c8a7f6bae231c53a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4e50377cd2357abf047fd9b0f540171a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7cbf9ea34c096a043368dc76a2781b7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.44804924726486206], dtype='float32').reshape([1]),
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_35d61ed0bdb4a095a6c81f2b82e67ac3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d06b55fc80b15a043b08edd6c60ae8d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([17571], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([17571], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d06b55fc80b15a043b08edd6c60ae8d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([17571], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([17571], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c5ae01231dc6d96bd61e5fce16208ec3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_13cfb9cd3a30341397a6a6cc9dcf2c2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ed7087b683a5bdd875cbe9f340dc0cd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 32, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_22957c795ee44dbea4bc5a63b8d95543(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26c496af661f594a452b0002b6269924
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 1, 9, 112, 112], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 2, 16, 9, 112, 112], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_49317fa304dc4897b66bc41120d87aec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_49317fa304dc4897b66bc41120d87aec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_49317fa304dc4897b66bc41120d87aec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_49317fa304dc4897b66bc41120d87aec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_49317fa304dc4897b66bc41120d87aec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_917e349e5a2a3ede1abd3d61e586fbcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([950, 81], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([950, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_917e349e5a2a3ede1abd3d61e586fbcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([950, 81], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([950, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8f0f49dd30639a87c66f23fe7d800b83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41816642619940f30c0c4565f2c485b9
    def get_inputs(self):
        return [
            paddle.to_tensor(-1.0, dtype='float32').reshape([]),
            paddle.uniform([950, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_917e349e5a2a3ede1abd3d61e586fbcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([950, 81], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([950, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b9fa18dafbd1970b116cd5718470458b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9b270820722d1ce09f79dbf16e9a9cc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 577, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9bb515bcfdf9a984177731b10b06a89c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 577, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b3fa11efb5c792af296c002bba448db5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ef905cefeca655659c06d8284090abaf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_af17129069581d160bda66762bd272ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef905cefeca655659c06d8284090abaf
    def get_inputs(self):
        return [
            paddle.uniform([768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 577, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_22add1c61857bba0bc1a9da68aca4989(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26c496af661f594a452b0002b6269924
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 1, 49, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_212bfb1a6f1d721ca743dd2ab0a89102(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 156, 40, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 156, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b73d33b20043d292f386fb153fe5e420(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_397da41d3099a2e2017789441028b595(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 52, 52], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc868e2208c87e84901b847049ec902e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([2204], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2204], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7f55dd5cd3383c90beb035fd18cb71e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_691aabc5536874dca095d7f1d739eebf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5644e648d02e8e1db3eff74a0c0c84e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5d018bd79d84fca4ff88da4b31f3749a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7cbf9ea34c096a043368dc76a2781b7
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.43094146251678467], dtype='float32').reshape([1]),
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_52b236b47c99943ee65533d6b2f3a9f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.143972247838974]], [[-0.1916181892156601]], [[-0.16192881762981415]], [[-0.36363685131073]], [[-0.3913705050945282]], [[-0.1610545516014099]], [[-0.5055286288261414]], [[0.5830764770507812]], [[0.295197993516922]], [[0.32580670714378357]], [[-0.35243040323257446]], [[0.42443472146987915]], [[-0.17313429713249207]], [[0.03083968162536621]], [[-0.1623837649822235]], [[-0.0004990920424461365]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c9c0afa68687a622713b125b014f6fe9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[0.47120556235313416]], [[0.46167635917663574]], [[0.46761423349380493]], [[0.42727261781692505]], [[0.42172589898109436]], [[0.46778908371925354]], [[0.3988942801952362]], [[0.6166152954101562]], [[0.5590395927429199]], [[0.5651613473892212]], [[0.42951393127441406]], [[0.5848869681358337]], [[0.46537312865257263]], [[0.5061679482460022]], [[0.4675232470035553]], [[0.4999001920223236]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_687b6d082b2a5795ff8cf59f87e550f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7d85bec64e71e98e6c1b2432a202fde7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7ccecca78825af7acba6e9bfd323581b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_16858515ad23d413b2d780cf110cc16b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([145, 336, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([145, 336, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3314381f30756a64ba0ecc482064cd1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5866308c4a5a2f459fda2beda2625178(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 8, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c7396c4e3c1825dc98203687e9ba5682(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.32693517208099365], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3314381f30756a64ba0ecc482064cd1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_15a0c8f6b5755a33d9588039258d0aef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ccefe6c71ebf4c16628ad6285fa95dcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f69f82faa7795e951184a81c407ba957(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 18, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_90152b3eddce526cf3a2768f7b2f04f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5fb3ecfac004bcd348b0ec418a251f9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bb65d1edca49c2f2c6c4f9012a5b297b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_efc2c158b1006f1fa94d51c5dad3c992(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_331a012e3e84aae2df4d7ad13b5c23f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 15, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2f3d107057a208985201404208bcab52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f509dbd7e0e0c9a27a06abcf61ad05a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_556970ff0c4177959751b18cedb70f45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 23, 41], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f509dbd7e0e0c9a27a06abcf61ad05a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a935011850fc5fad8f607b84ffecacc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 46, 82], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f509dbd7e0e0c9a27a06abcf61ad05a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0e56bc73d03ebb0588b25b8ca7e9cdfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 92, 164], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f509dbd7e0e0c9a27a06abcf61ad05a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_bbf1dafdedb1757d63408d7d2297a723(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 184, 328], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f9fefe93ee8f3b956ef01bb52a3c36da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[10.327248573303223]], [[-17.276676177978516]], [[0.9485893249511719]], [[-16.85544204711914]], [[7.114154815673828]], [[-7.70201301574707]], [[9.863938331604004]], [[16.699020385742188]], [[12.297445297241211]], [[8.553190231323242]], [[11.113935470581055]], [[17.291950225830078]], [[-20.813465118408203]], [[12.612366676330566]], [[-11.822504997253418]], [[-2.9844284057617188]], [[2.054971694946289]], [[1.585641860961914]], [[4.797857284545898]], [[-2.8156657218933105]], [[8.280933380126953]], [[10.853103637695312]], [[18.358356475830078]], [[7.910577774047852]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9617d1ed1df4a6989ad2386d543336de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 23, 41], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[1.0]], [[0.0]], [[0.6897178888320923]], [[0.0]], [[1.0]], [[0.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[0.0]], [[1.0]], [[0.0]], [[0.0]], [[0.9109943509101868]], [[0.8171284198760986]], [[1.0]], [[0.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_418946df61c06ecdace638a17ec1953a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[23.25609016418457]], [[22.068134307861328]], [[-4.4802961349487305]], [[21.55982208251953]], [[20.06050682067871]], [[-2.3278143405914307]], [[-10.545915603637695]], [[13.066387176513672]], [[-12.669084548950195]], [[-6.823731899261475]], [[5.274051189422607]], [[0.0919073224067688]], [[-8.944574356079102]], [[34.21359634399414]], [[-10.43765926361084]], [[-10.930832862854004]], [[-14.796675682067871]], [[-22.209365844726562]], [[10.01297378540039]], [[3.0613644123077393]], [[12.429766654968262]], [[-3.6083779335021973]], [[-1.5522977113723755]], [[-4.993011951446533]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f94dc09754cc61118ca77559e9ef7836(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 46, 82], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[0.0]], [[1.0]], [[1.0]], [[0.03443711996078491]], [[0.0]], [[1.0]], [[0.0]], [[0.0]], [[1.0]], [[0.5183814764022827]], [[0.0]], [[1.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[1.0]], [[1.0]], [[1.0]], [[0.0]], [[0.18954044580459595]], [[0.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_1df25220a5d9484c266dcab98e2f86ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.640294313430786]], [[-12.785752296447754]], [[10.169280052185059]], [[-13.4395112991333]], [[19.346017837524414]], [[-16.668350219726562]], [[-20.690399169921875]], [[-20.79280662536621]], [[6.6085405349731445]], [[5.270244121551514]], [[-21.610952377319336]], [[13.067865371704102]], [[-4.982135772705078]], [[-5.352970123291016]], [[-6.628631114959717]], [[-3.453378677368164]], [[-10.739202499389648]], [[-14.640015602111816]], [[-28.33976936340332]], [[10.556808471679688]], [[-4.7640767097473145]], [[-7.026877403259277]], [[10.328776359558105]], [[-5.711918830871582]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8c0eef43f309c6e5aabf50bf76c62183(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 92, 164], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[1.0]], [[0.0]], [[1.0]], [[0.0]], [[1.0]], [[0.0]], [[0.0]], [[0.0]], [[1.0]], [[1.0]], [[0.0]], [[1.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[1.0]], [[0.0]], [[0.0]], [[1.0]], [[0.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_cf25a61fc4628723f556134f73c2b789(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-1.6406688690185547]], [[0.11720094084739685]], [[-0.7195172905921936]], [[-0.2728418707847595]], [[-0.027154400944709778]], [[-0.8047922849655151]], [[-2.0279572010040283]], [[-0.6226431727409363]], [[1.3061802387237549]], [[-0.31873783469200134]], [[-0.6287018060684204]], [[-0.7890415191650391]], [[-0.04433278739452362]], [[1.4381437301635742]], [[0.5753604769706726]], [[0.4178016781806946]], [[0.2388555109500885]], [[-0.7774878740310669]], [[-0.6769956350326538]], [[0.3888126313686371]], [[-0.4195915162563324]], [[0.530755877494812]], [[0.21144956350326538]], [[0.9624615907669067]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0b6c000ad231cc8d0e3d1e886c8292db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 184, 328], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[0.17186620831489563]], [[0.5234401822090149]], [[0.3560965359210968]], [[0.4454316198825836]], [[0.4945691227912903]], [[0.339041531085968]], [[0.0944085419178009]], [[0.3754713535308838]], [[0.7612360715866089]], [[0.4362524151802063]], [[0.3742596507072449]], [[0.3421916961669922]], [[0.491133451461792]], [[0.7876287698745728]], [[0.6150720715522766]], [[0.5835603475570679]], [[0.5477710962295532]], [[0.34450241923332214]], [[0.36460086703300476]], [[0.5777625441551208]], [[0.4160816967487335]], [[0.6061511635780334]], [[0.5422899127006531]], [[0.6924923062324524]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_039a367aabb9c652a3120064b9001287(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([1.079062581062317], dtype='float32').reshape([1]),
            paddle.to_tensor([0.32380127906799316], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d6d196547a107893f5ff6848bc04aa22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0859946012496948], dtype='float32').reshape([1]),
            paddle.to_tensor([0.12816721200942993], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_869e7ae003532583d2348386c304a493(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([1.4620434045791626], dtype='float32').reshape([1]),
            paddle.to_tensor([0.0850069522857666], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_79e802e7f6ade783ea90b5b7f2f85025(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0012603998184204], dtype='float32').reshape([1]),
            paddle.to_tensor([0.2681646943092346], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8d28b4794dfb218b03bd7ae30cb99288(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([1.5395174026489258], dtype='float32').reshape([1]),
            paddle.to_tensor([0.3222275972366333], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f67d21ea99abb5315218baee34c1a737(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.642633318901062], dtype='float32').reshape([1]),
            paddle.to_tensor([-0.18892067670822144], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_30c085c15a52f66c06d64c2d2e2de145(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([1.3673673868179321], dtype='float32').reshape([1]),
            paddle.to_tensor([-0.453180730342865], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fbc1aab7a2c60abbf0c4c3dacb3437a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([1.1933609247207642], dtype='float32').reshape([1]),
            paddle.to_tensor([-0.2269984483718872], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_22f981dfc9822fd414421e783c872ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.6406524777412415], dtype='float32').reshape([1]),
            paddle.to_tensor([-0.14665278792381287], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d15804ea81f9f2e2ec7d50cd1331ebd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 9216, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a51395e0b89448221529248f3d87a068(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 9216, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_41656768658f72557eb1b186ae2b013d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_672d1e3bc7f3800a8119d10b23400b2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_efc2c158b1006f1fa94d51c5dad3c992(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3d949e628975f08998e8916140202b82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 13, 13], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fa767aceb33df9fcb152fa73f720cc82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.33148193359375], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_570d1038ca772afff599a3548bb618f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0b80d57590ad41e7f890d3f604280db2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 784, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1e67694fcf2d31455cc375b606a87135(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_28e70d6740dc6404c75ba5563a0d8aca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b3b9348a4a596d775530faeb0e6812e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b38c54914b248cf4219c062f264f281b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_fb6407e69c4aa1565ef919c41a2cefc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 30, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ef9025d6249cc539bd6dd1de707c0419(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 160, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a9335357df1b2dab018f7c278c041dc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 160, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ba89b3d64e03e6d3f1b2a9d9568438bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2ec0600e2c0d6b2a17a37b6ea28691e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.028895676136016846], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2ec0600e2c0d6b2a17a37b6ea28691e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.028895676136016846], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_09eab07d3341cdd8ae9136053ff9af21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_67fc969134571bd21e0b0b8d312803bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.29860013723373413], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5a2de620b6d5b8dfa2bd8fe89555a496(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_60879054237e31181ba31ff05b1b922c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 104, 104], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2f30d5da65d2a909682686ceb42c9f64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([16384, 5], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_472a7b87e9c4bd9112b14b028b27ffc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_469ed82c102892fe0dadc09aef842490(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 104, 104], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_20c107a070b99279e4832b3b905cb66e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_10e9ad1f3e16c64500358a2cf873c22d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_13785115db9a95a2420c0e44c2676c1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3f4f8eccba6b9722be4d0272edd45d41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5a6de823ed0a14155a7289f3a17f0ad4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 40, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_34d2ed6168734195f79b4e4af4f6ffc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_34d2ed6168734195f79b4e4af4f6ffc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_34d2ed6168734195f79b4e4af4f6ffc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_34d2ed6168734195f79b4e4af4f6ffc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6a1dc02ac9aa61dc3f6c050e283c2491(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4a3eaf87d17739314cc6d1fe16d9e9de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4f400bf9b34cae73b9cf896ca6f16867(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([70, 81], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([70, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4f400bf9b34cae73b9cf896ca6f16867(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([70, 81], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([70, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b06ca5b4b7512a6c69215fde4a8b6184(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41816642619940f30c0c4565f2c485b9
    def get_inputs(self):
        return [
            paddle.to_tensor(-1.0, dtype='float32').reshape([]),
            paddle.uniform([70, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4f400bf9b34cae73b9cf896ca6f16867(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([70, 81], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([70, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2e276e775665607b4e0b0184f3ab2b0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 80, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.4576486349105835], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_44cbd50f489b01e3bdfeb0a08c92c887(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([6, 144, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6, 144, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d5063f9f33dbc99db6501d496dc874b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([6, 144, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6, 144, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0994c115248fd6b73abf7de22fa153e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([6, 144, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_472a7b87e9c4bd9112b14b028b27ffc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f5cf893f6896e4a0c3022e80ff731c55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 184, 184], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_28e70d6740dc6404c75ba5563a0d8aca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6b4990a02d115ccf1392b6fefe7b1d75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 34, 34], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b33ca3ed7bd30b54dd2402666271ae9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([15200], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f509dbd7e0e0c9a27a06abcf61ad05a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5f64781023a0f92a59c7a53c54e6c08f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 20, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5eb41ec25da73f1749b5657f5972c8e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([22, 480, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cec9761a0237c5ae82cd94ab7a9c157a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 872, 20, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 872, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0e66f5bd2c5458c016b3f9caef8ac217(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3465f94328d338b41b548d554fac9354(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 22, 22], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2911daf1003eff246a6e8c011777bef1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([150], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([150], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9f4a8ce8fd8f549608a59f0962ace51c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9f4a8ce8fd8f549608a59f0962ace51c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9f4a8ce8fd8f549608a59f0962ace51c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9f4a8ce8fd8f549608a59f0962ace51c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9f4a8ce8fd8f549608a59f0962ace51c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_93df42fb4f3b60696854b9a6eb28f956(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_cb148e54222c95b325da881ffad542f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 60, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9518e7ecea8f39cbaab241c1c775b45a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_87e6ecab9a7ac67e6e5586eaa1f1dfad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d635edf2a5ca15ce678bae548f22adcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6977641444720505416d38c00b61e03a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_df17922f099f7cf3ef6b5eacb138c730(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2041b6946153d6a3fdc3584c6125113b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 320, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fddad8de81aa2b059aa3f7a23b3ffca1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7c2355798bf7d49cc5c80f27bffb28e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_04c68db8686049d6a4e7705c9f9a8d33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f31cde4239d56e97e1f617f2afde02c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bbce7c59d57c58cd1c295fd175558070(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26c496af661f594a452b0002b6269924
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 1, 49, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_472a7b87e9c4bd9112b14b028b27ffc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3b615fd66ecf5b93cfe8518f226a714a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 76, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c8e1cc4ae6693cd0a02be72cff0a9878(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c8e1cc4ae6693cd0a02be72cff0a9878(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c8e1cc4ae6693cd0a02be72cff0a9878(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c8e1cc4ae6693cd0a02be72cff0a9878(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c8e1cc4ae6693cd0a02be72cff0a9878(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_05bcd2a3ff6810ddd6e896efefeeb29e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 872, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 872, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_20c107a070b99279e4832b3b905cb66e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_10e9ad1f3e16c64500358a2cf873c22d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_13785115db9a95a2420c0e44c2676c1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_114fed18848b54bf74dc12af1cb833d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_182e5be9a78d93d86dc49cf8d001202a
    def get_inputs(self):
        return [
            paddle.to_tensor([4204], dtype='int32').reshape([1]),
            paddle.to_tensor([4], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_0b9c63ab974c060caded2483461945db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0b9c63ab974c060caded2483461945db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0b9c63ab974c060caded2483461945db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0b9c63ab974c060caded2483461945db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0b9c63ab974c060caded2483461945db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_132daf71fe91c1d82bf9b575c7a08031(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([4204, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4204, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_132daf71fe91c1d82bf9b575c7a08031(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([4204, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4204, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0b9c63ab974c060caded2483461945db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0decf1ab0797f1ebd880929ceedaa96e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ff51f0ce898a0af096e937c03778d846(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c5ae01231dc6d96bd61e5fce16208ec3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d08412c148e7b36600d54a7de5c08aa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 256, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_472a7b87e9c4bd9112b14b028b27ffc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5ae90e26a00c1ae3401d7c14d12d26fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f9766251d6d4f227cd5bc1acde9ea521(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b0b4f456bf0dca5bd1cabfe4acd7e872(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ad1b91ea104d7c603cbac75200700d74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_337c23fcb3c6108117d04e547be463ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_337c23fcb3c6108117d04e547be463ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_337c23fcb3c6108117d04e547be463ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_337c23fcb3c6108117d04e547be463ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_337c23fcb3c6108117d04e547be463ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_03841210a13d8bef89ceb15e2ed15155(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a6a6f59c446a8f0aff3443ca04ad5b68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 9, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bdf21a8ee7a6441046a162878f77082a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e44718143120f147e1a39cfe77289194(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3314381f30756a64ba0ecc482064cd1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a9055a8d03c51827a7e212da74ca54fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 30, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5a2de620b6d5b8dfa2bd8fe89555a496(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9b33d20cc4e30256cb7298bab8a9504b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 48, 48], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_410b806830932e0595f0399f308ced83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([551, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([551, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_410b806830932e0595f0399f308ced83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([551, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([551, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_410b806830932e0595f0399f308ced83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([551, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([551, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_81741f96a2bb8f9c975a5e080c96a71a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_30936c4b0732a526508880ecc8fee74c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4096, 5], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d635edf2a5ca15ce678bae548f22adcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_633db62740f60a715e4a94dc87da858e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_90152b3eddce526cf3a2768f7b2f04f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5fb3ecfac004bcd348b0ec418a251f9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bb65d1edca49c2f2c6c4f9012a5b297b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_93fe074755f71d7e15cf73843702fdf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([40], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ee8aae19a8c6f95da76bb73faa646174(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 92, 40, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 92, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0decf1ab0797f1ebd880929ceedaa96e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_876c0f3d9e9b33ff2cf978025befb784(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 30, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_63b984a7ada9e47e37140b8dba69b858(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4f111f1a4daeb2034f6a91c2af3cf813(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_836ad19880c7e90f3b7762a3a240f452(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b065aa9e7e79bb22865f32620221d0f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5558e9fd0ded78e52853e878f0488241(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f8e4667cf89a82c69a4ce4a943f4afe5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([22, 36, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 36, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9f1a30f156fd49430f3982bcf0b28e50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3800, 81], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3800, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9f1a30f156fd49430f3982bcf0b28e50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3800, 81], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3800, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1844758c09346c6247a41d56eb822678(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41816642619940f30c0c4565f2c485b9
    def get_inputs(self):
        return [
            paddle.to_tensor(-1.0, dtype='float32').reshape([]),
            paddle.uniform([3800, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9f1a30f156fd49430f3982bcf0b28e50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3800, 81], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3800, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_03841210a13d8bef89ceb15e2ed15155(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_008cca95c857c2131507fc64ce84d5c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 13, 13], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3314381f30756a64ba0ecc482064cd1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_36f20170043a50397e36560928bb6bd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 92, 92], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_df17922f099f7cf3ef6b5eacb138c730(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2041b6946153d6a3fdc3584c6125113b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 320, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fddad8de81aa2b059aa3f7a23b3ffca1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ac60da5eea5ff89b2fabc6effea6cfaf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_eb3f607579e48ddd730351c54661349e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_304992211c84782c37658b30183445e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([4, 144, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 144, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_57d11bdd6474b278c11559fae1abdd68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([4, 144, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 144, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9a8eb92e9e086aa030795a2dbb846305(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([4, 144, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0884721cbea88713d0248b06bbeadc33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ec2e39e7c4d5371f82289ef4beb0d7a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.4766753315925598], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ef6810171e12c271503a6159fcac0d5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_721de4bd00896386f9a8f77391f4dac3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3314381f30756a64ba0ecc482064cd1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_48f8b752683cae834110ec703867030b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 20, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3f8997982c442597600c38d7ffeeb73e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([3800], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3800], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2b9943ee3b6f4b1aad04460f2fa83ff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_182e5be9a78d93d86dc49cf8d001202a
    def get_inputs(self):
        return [
            paddle.to_tensor([4680], dtype='int32').reshape([1]),
            paddle.to_tensor([4], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_4c6bc2d9dafb6a7951cead8919ba873f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4c6bc2d9dafb6a7951cead8919ba873f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4c6bc2d9dafb6a7951cead8919ba873f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4c6bc2d9dafb6a7951cead8919ba873f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4c6bc2d9dafb6a7951cead8919ba873f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_98eebd844663b2831ac50a0d79efb5d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([4680, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4680, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_98eebd844663b2831ac50a0d79efb5d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([4680, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4680, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4c6bc2d9dafb6a7951cead8919ba873f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_82c77d02624bad34939c0c1e5ba87610(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bec6dca242fcdf4a7763a9c1bf6ce7bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_95c4723c59542362e4784122afdd492c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2832049614fd4273c0901f1886a24b5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3bc5c2142495ff854c0faec3cc62fe8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 784, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a5f42fbf3f89d2a7890d8fab32858fc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_836ad19880c7e90f3b7762a3a240f452(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b065aa9e7e79bb22865f32620221d0f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5558e9fd0ded78e52853e878f0488241(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d764c137319fe6fb6f84460e541f19e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([171, 36, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([171, 36, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c7b5bf0cf323552abffe7da62d4ea71e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_182e5be9a78d93d86dc49cf8d001202a
    def get_inputs(self):
        return [
            paddle.to_tensor([3778], dtype='int32').reshape([1]),
            paddle.to_tensor([4], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d24d8c24d422bd93b25608973bcc5a8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d24d8c24d422bd93b25608973bcc5a8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d24d8c24d422bd93b25608973bcc5a8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d24d8c24d422bd93b25608973bcc5a8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d24d8c24d422bd93b25608973bcc5a8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_daa4aed0d8555998086b0ece3e2a3785(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3778, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3778, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_daa4aed0d8555998086b0ece3e2a3785(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3778, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3778, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d24d8c24d422bd93b25608973bcc5a8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ae723d1f4e2a0af986110d50ff3a1e16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d66569920e5031c31a8c849401aa1471(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 100, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5f566e56ae9df223c4d0d17844ec0901(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4cbc398c65cf7ae8e06d52ba64b2835b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 1248, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1248, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5f5d1ee7a72b78cbc78ef6efd8749dda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a82831d136b6a5ac8f5d6e99a7fbc474(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_516ee1da59c69f3a174bd5922c12e2d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[0.5635396242141724]], [[0.4792746305465698]], [[0.3874158263206482]], [[0.6098951101303101]], [[0.4213833510875702]], [[0.49063026905059814]], [[0.45961010456085205]], [[0.5237249135971069]], [[0.6601620316505432]], [[0.6455176472663879]], [[0.45134130120277405]], [[0.37797337770462036]], [[0.4350909888744354]], [[0.5512875914573669]], [[0.49409928917884827]], [[0.5079438090324402]], [[0.48372700810432434]], [[0.37882086634635925]], [[0.4952718913555145]], [[0.610732913017273]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_da61655ab478751c3f9b2bd109fd35aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_12b5bd3e1c0fe19c71f70a69d5d86ac6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ccba10543970b37891be3c8020af255e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1174, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_89ac87d4af6d706e8cd4ad5604a491fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1174, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d887a91f60623237cbd22efe1b8b6eaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_407a9ff796d5cf2f8011d10f7742b6a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_407a9ff796d5cf2f8011d10f7742b6a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_407a9ff796d5cf2f8011d10f7742b6a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_407a9ff796d5cf2f8011d10f7742b6a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_407a9ff796d5cf2f8011d10f7742b6a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_410b28c531cee4771f748e1542d0aa20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0a7cef7bdf5e56083059db69a6078965(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([22, 336, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 336, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a5fc6fdc2afd82c15888046b82ae8cd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([145, 240, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([145, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3314381f30756a64ba0ecc482064cd1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5f8ee783482d634da64efade51f9689f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 48, 48], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d8d2881e1667e44e002d54b9c13ea98c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([171, 336, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([171, 336, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3314381f30756a64ba0ecc482064cd1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5ca77c93277b392cad16066fa6aa616b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 38, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b892123857428b0dff8a3bde2a723724(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d47107b8302089108238929c576ed55e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c72cb575f3c1974c9459dab9e0bfdfa7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bdf21a8ee7a6441046a162878f77082a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5adaf9044c45ab6f6102cbd81b0368d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_888801390788289605d10b7cd81591c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d57d7a35b9818974cc0dd6e6ade0def8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5a2de620b6d5b8dfa2bd8fe89555a496(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6ccaad95b751bc3ee9740cb0c7bed838(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 152, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_96711fb41d1625a504205d8d16d8edbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([4, 576, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 576, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6c080a4310d964cea7af69123f012629(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([4, 576, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 576, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eb55742b0dd6d0686571a5f240fc6030(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([4, 576, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5d6b053d8e47cb09c6a930be5d4145f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([171, 336, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([171, 336, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d1ca3ecb1f562447c97635b25c2afcb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1024, 5], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_888801390788289605d10b7cd81591c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_828c6cb62cc9a9be79c5554c18eee7be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 22, 22], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_73e6e8a7b0252e41e0554920e5c0e6bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.4415740966796875], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8d568f54870b0fb1b40a9ec02a82d17e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8d568f54870b0fb1b40a9ec02a82d17e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8d568f54870b0fb1b40a9ec02a82d17e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8d568f54870b0fb1b40a9ec02a82d17e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b6ceef544aef363bbbccc049556e5093(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8f66b0d094499062c13d3e42ebd80d23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_37cf92e6a5a333690b8e9be4b250cec6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26c496af661f594a452b0002b6269924
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 1, 49, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 4, 16, 49, 56, 56], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_00b9e74919fdf6f4a27ff636669b12aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ebe7487d9934cf921b4e681516e0553d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 60, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_93df42fb4f3b60696854b9a6eb28f956(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a9982fe0949dceb3eec51061f1921f82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 48, 48], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9e6044f4b2213baea05ee64913825c89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([10, 480, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d88923f965106b39924a4d6b5fc1dd00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_472a7b87e9c4bd9112b14b028b27ffc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_410b28c531cee4771f748e1542d0aa20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_66878a721703762289ecfa05a9ba1922(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_66878a721703762289ecfa05a9ba1922(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_66878a721703762289ecfa05a9ba1922(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cbafaacf096bdf48e444f8d7f52a8a3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 7, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.46113526821136475], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a7906edc6d7176413ee7f69642c5c173(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7da09e0ba6fa50e13d6b25451f70aae6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b892123857428b0dff8a3bde2a723724(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d47107b8302089108238929c576ed55e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c72cb575f3c1974c9459dab9e0bfdfa7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7964c34fba9bc3bcd0584b019be5bb76
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f13daf3f449b691a958a3282c3b2a4e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_fca62ad0b9f39a8e689e4a0db428ac23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f237d32478dafa581625b663f42810a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([247, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([247, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f237d32478dafa581625b663f42810a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([247, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([247, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f237d32478dafa581625b663f42810a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([247, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([247, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1aa6928067144a49fc9e2ec5eac36f29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9ba088776f000302d0249d450d1b0b36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_efc2c158b1006f1fa94d51c5dad3c992(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b013c8535fd03d8c9d70c6b19520e3b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1edcc826d55a1a8ccf8d2a7cc6c0660e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([247, 81], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([247, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1edcc826d55a1a8ccf8d2a7cc6c0660e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([247, 81], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([247, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4b6e3994690fc08253006a9309d5833e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41816642619940f30c0c4565f2c485b9
    def get_inputs(self):
        return [
            paddle.to_tensor(-1.0, dtype='float32').reshape([]),
            paddle.uniform([247, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1edcc826d55a1a8ccf8d2a7cc6c0660e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([247, 81], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([247, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_687b6d082b2a5795ff8cf59f87e550f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_85e3665bb7659debbbc665b27a4de5a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 15, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eddb038dad90ad66ccf9a4fcc2a44fd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64115a9b452c8647c7f4daf6bf43008
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_df338a6f7a082ebfa006226b416a3600(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 18, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8c915e9df55eca30e3996971b13d346e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 92, 140], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.23427656292915344], dtype='float32').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()