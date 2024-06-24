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



class PrimitiveOp_f29d61a42275d6a4289b8be3acc810c8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 15, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_118a9be2ae1b1e1e95d791bb9dbc0206(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29d61a42275d6a4289b8be3acc810c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_06f827971a8f84f612d65cc80006e776(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_23d6fefdecb5d88e4dd8557218f63671(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06f827971a8f84f612d65cc80006e776
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d0617dcf33c4fe3b1979074e17e5dbbd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 91, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_017470034610b94343ae0c0b5333fa97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d0617dcf33c4fe3b1979074e17e5dbbd
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ef6eca8324cb89c425ead0be71e45bdd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 784, 6, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_414776fa0a6a5de36682f642a2df1367(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef6eca8324cb89c425ead0be71e45bdd
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 6, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_eb9f70282899345810179c07997ffef4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 784, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_98a4dde342bc73ca96d006e07e9af545(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb9f70282899345810179c07997ffef4
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4dc2635fc07050ef006a0bdd589c884e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 192, 49], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e5708d33f552d9b77709714e997a74d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4dc2635fc07050ef006a0bdd589c884e
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 49], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2945ca61ad8e249c717d30897e1fd40f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 2, 6, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c6d1b02dfe21bbca641d45bc75fde578(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2945ca61ad8e249c717d30897e1fd40f
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 2, 6, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_765dea6d119b72d25e404af98eb9718a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 6, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_27adff2a34754b179e250911ce9593af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_765dea6d119b72d25e404af98eb9718a
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8512bd3114b7230ac053059ab21a7909(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 168, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bc82e2907340c3ad7c69b0b3fda4f1ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8512bd3114b7230ac053059ab21a7909
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 168, 256], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_13efba799831ba1aa390311a4f08c4fa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 84, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b2961b7ea680463d39e90ed1ada8ad1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13efba799831ba1aa390311a4f08c4fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6e3c1d70017703f35d3bbbb2396cac84(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 42, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cc0694287df16983f47af8ad706262a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e3c1d70017703f35d3bbbb2396cac84
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cb275b34ab43129a93a2f7c5477f078a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 21, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_60431b7cfd512c1d4311822fe6975d63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb275b34ab43129a93a2f7c5477f078a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4f8f98bda767ec9a08d1da019e7a1692(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 11, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_deef41e3f816968baebc657eb4759f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f8f98bda767ec9a08d1da019e7a1692
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_66324211933c13de12f9da59eae6762d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 168, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_76c0b9c7c130a1dd5275cf2adee7b4eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66324211933c13de12f9da59eae6762d
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 168, 256], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c1cd6e6deff5cf978bbe15ef302ff410(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 84, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a16f2139c3442a5ce398b72b9fb4d6fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1cd6e6deff5cf978bbe15ef302ff410
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 84, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2afa636170cc821e253a1fb2761f2c61(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 42, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b9c546d508f74c106221360d1e62578d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2afa636170cc821e253a1fb2761f2c61
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 42, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8e489f1e99d59a61126b341ac3ae44df(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 21, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d74a95729b32bd1638ec277bffb8b2f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e489f1e99d59a61126b341ac3ae44df
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 21, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a1144ac8a6473d933300e925b1b922cb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 11, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1d97a2715657d571c415936c2987dead(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1144ac8a6473d933300e925b1b922cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 11, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1021a233ed67b23276c8f0415e98e21d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300, 256, 49], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cdbda394e96ed1a5cac707ab2fe4e4c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1021a233ed67b23276c8f0415e98e21d
    def get_inputs(self):
        return [
            paddle.uniform([300, 256, 49], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bb7f956137ad38cc22acf3dc633d470a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 8, 7, 8, 7, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b378a7e57743b4e61da2407d39e0c526(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb7f956137ad38cc22acf3dc633d470a
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 7, 8, 7, 96], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_76cefa5e02fef6152f89c889e46792ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[3, 0, 1, 4, 2, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 64, 49, 3, 3, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e586202c5c1056723578cea25de3b353(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76cefa5e02fef6152f89c889e46792ee
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 49, 3, 3, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_68bdbe2c601197f5a6528c3b85cf0f2b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 2, 4, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 64, 3, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3791670b2db9fee4ac81f89e2ff20c77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68bdbe2c601197f5a6528c3b85cf0f2b
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b4590d695db9f8983f1bc26345670654(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 100, 3, 4, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e2e2307ca0e823e177d317d8479f8a9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4590d695db9f8983f1bc26345670654
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 3, 4, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7c6da0f0264d7892c1de7fb8d258e236(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4, 100, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_00294c4b9aa581ba63722bb4d9a5c9df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c6da0f0264d7892c1de7fb8d258e236
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_138bae9e0cfad9b1e553a4cf0b6144a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4, 100, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ec2a1073a4bf822e37ab93e0566c419c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_138bae9e0cfad9b1e553a4cf0b6144a3
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2b26024e6f896588df7a7f7bd5127c0a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 198, 3, 3, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_07d2ace0e627f3076244aafcf8380216(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b26024e6f896588df7a7f7bd5127c0a
    def get_inputs(self):
        return [
            paddle.uniform([54, 198, 3, 3, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d1f27af9693ebf06f61ba32fb9cc3053(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 3, 198, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_af0ff8bfb7d7c400ab66289c57eb9f4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1f27af9693ebf06f61ba32fb9cc3053
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 198, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_92c72fa054e382305af6e24ebc5d08bc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 3, 198, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b44de019964920cd22ba2c6c9e34b41f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92c72fa054e382305af6e24ebc5d08bc
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 198, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_397302b126ccdacea9dde585e1c4345d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1960, 16, 2, 4, 6], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_99beaf06630fe0ca73b463532c9e95e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_397302b126ccdacea9dde585e1c4345d
    def get_inputs(self):
        return [
            paddle.uniform([1960, 16, 2, 4, 6], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7c15f52b3d29ea5aa27b47b54514c4e1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1960, 16, 4, 6], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a3f9ed83dab382af4b43371c4f51c93a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c15f52b3d29ea5aa27b47b54514c4e1
    def get_inputs(self):
        return [
            paddle.uniform([1960, 16, 4, 6], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f9c074e871e9dd2fa48802a463e7d808(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1960, 4, 16, 6], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c89f0085f2de051aa1ec7dda061ce268(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9c074e871e9dd2fa48802a463e7d808
    def get_inputs(self):
        return [
            paddle.uniform([1960, 4, 16, 6], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0264b42ee0b22b0800ba3f737b6b7897(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 784, 6, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3ec88320ea082b5e15abc7703263bfae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0264b42ee0b22b0800ba3f737b6b7897
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 6, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_76dee8755d5dd20cf8c34a12ce9eb7c2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 784, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d08fb43735e1af17a9cbe60dce5f95dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76dee8755d5dd20cf8c34a12ce9eb7c2
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_60cb79dee0cb1e9fe788d25e98edb957(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 192, 49], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8be14f6c791d87f70b673b3048873f49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60cb79dee0cb1e9fe788d25e98edb957
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 49], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_960222d2177659dd07d97e15a0dc249b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 2, 6, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ba9b2baeb7b0323e9e76de8c56fcca0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_960222d2177659dd07d97e15a0dc249b
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 2, 6, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8718f08430db77a6af404763fce7563f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 6, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f65f263ecaf0bd0a7f650ac3436835c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8718f08430db77a6af404763fce7563f
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1dfcb3800fa9bf8954a8bc7de5fd8026(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16, 32, 128, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a20ede25e21a3c56cc905a4f61a5091c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1dfcb3800fa9bf8954a8bc7de5fd8026
    def get_inputs(self):
        return [
            paddle.uniform([16, 32, 128, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d923aa8a380eb500fd9c3aca62f98761(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16, 128, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_104e49c8f54daea67bd3750a6ff997cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d923aa8a380eb500fd9c3aca62f98761
    def get_inputs(self):
        return [
            paddle.uniform([16, 128, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_51b56d5ecdf807a71e270ccee0eaa97b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 7056], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a630298ed4489dcacdb42abcf6547af8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51b56d5ecdf807a71e270ccee0eaa97b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 7056], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2a14be703bcfeebda8119ea0c70cdab3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 7056], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8457af051f4b4057db24763d7a1d0066(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a14be703bcfeebda8119ea0c70cdab3
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 7056], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ef4be89125fd8bfec5dde5279597f614(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 8, 7, 8, 7, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_343e76464d282b75911f1e27a5b17fdf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4be89125fd8bfec5dde5279597f614
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 7, 8, 7, 96], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c09db675753c27b512b2ff6d36a883ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[3, 0, 1, 4, 2, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 64, 49, 3, 3, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a1ef02ae7cb80742dda80c29c9e03484(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c09db675753c27b512b2ff6d36a883ef
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 49, 3, 3, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a821b35effdac8f38776368a78c6e04e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 2, 4, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 64, 3, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_39baf3381351fc67df513864b7d8ae3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a821b35effdac8f38776368a78c6e04e
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1053c0ff64f9d89fc24745add30b0f83(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549, 4, 19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_83fdc869c4dd59e749911b383bb5f515(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1053c0ff64f9d89fc24745add30b0f83
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4, 19], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d33ebe7eb2350c6a5f7fbef046cdb950(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 160, 240], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_20615454e66cecddc9207f7e65b2f3cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d33ebe7eb2350c6a5f7fbef046cdb950
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 160, 240], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3b2b3755fd6cbef82e8e4c40e1d79cc1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 80, 120], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d96486291b4989da1d3d0c09ae42e2ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b2b3755fd6cbef82e8e4c40e1d79cc1
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_025388fd9e00346610a5a8d6bf4e6005(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 40, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fae30bdbdd0dfaedf69d6c50ce001e50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_025388fd9e00346610a5a8d6bf4e6005
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7b4f5c90c6fd97c7072322c3f159a1cc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 20, 30], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_84b305a17fd24392ba184f8afe00301e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b4f5c90c6fd97c7072322c3f159a1cc
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_be85d04100b559b512b10ddee033a188(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 10, 15], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ed59af20cc0a93c657293c06d98c5e53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be85d04100b559b512b10ddee033a188
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_41fe3a6483ff0ad187409d1b1e6f5484(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 160, 240], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d83d3f3470bd56a59f9c112a6c1954b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41fe3a6483ff0ad187409d1b1e6f5484
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 160, 240], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_03ff2fa85203d463c5bbf5e746cd5428(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 80, 120], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_759f9bf76abf618d3690c017fdc3d1fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03ff2fa85203d463c5bbf5e746cd5428
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_914169f7edb2ee922ddf52d5f3c38163(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 40, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_63ced54a8f8b55a4de8a48dee2431641(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_914169f7edb2ee922ddf52d5f3c38163
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cce98897e476af4be17fc26091f3ba1b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 20, 30], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8938ac688571909d2d22e61705e24b23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cce98897e476af4be17fc26091f3ba1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_799b315ebdce0326635ae59331852079(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 10, 15], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ab50a6bbddc48a95d542f05fe02816d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_799b315ebdce0326635ae59331852079
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_19267aef56f678aed366fb18810c7bf5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4a4d3116c84341e74b589055280186c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19267aef56f678aed366fb18810c7bf5
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 256], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_231e35d7454c3cf4c2eb931569b05d13(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_91598069af5ebb6c96083c2b7d75476f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_231e35d7454c3cf4c2eb931569b05d13
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 256], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bfa4a73154cc9fd661fde3f846f97d67(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dd89db0d9838a3cb734728570c5b62c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfa4a73154cc9fd661fde3f846f97d67
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 256], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_581351d2c9f222d4e806d2f588dfd829(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 576], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_470489840a6126a06e53c706c1a1146f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_581351d2c9f222d4e806d2f588dfd829
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 576], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e6f4128b5aa91d80541ab4f5fcb6f9e9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 576], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6bbe1fab07a2d8b77bf84006036cd112(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6f4128b5aa91d80541ab4f5fcb6f9e9
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 576], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_112b2548816103019986f69677d7b728(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 2304], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_645fb0c2c28c847a8530df8cb42db822(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_112b2548816103019986f69677d7b728
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 2304], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_291f7e7c973ea004a299570d6cd22ca0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 2304], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2679ef5e7b61beb1ce38752d40d6a686(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_291f7e7c973ea004a299570d6cd22ca0
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 2304], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cd471f5a8daf7f77a06ef2ddde2f493c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 225], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4b995b2ac60b12588c12bb9c8d44c2a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd471f5a8daf7f77a06ef2ddde2f493c
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 225], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_57373ea12f862db5375fc3811a66cc15(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 225], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9f69499672c95ed61547fe9fe6a26985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57373ea12f862db5375fc3811a66cc15
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 225], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e2420ed4590915e918010772c890c36c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 256, 49], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5da5002f6ae0fa2d838355debc603a69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2420ed4590915e918010772c890c36c
    def get_inputs(self):
        return [
            paddle.uniform([100, 256, 49], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4e8d9b63e2c91b585c5fed809877a683(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 20, 128, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_612087630e0dcc2c72af6775c3d9003f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e8d9b63e2c91b585c5fed809877a683
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 20, 128, 256], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_892f8c36460cc78452c2435bef822a26(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 40, 64, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_719b418e1a34bf6366cdcd24fbeb1154(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_892f8c36460cc78452c2435bef822a26
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 40, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c61d9a0d4783603720a717a2cec2850e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 152, 272], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a31e2055487d3ad3384d32d3294976a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c61d9a0d4783603720a717a2cec2850e
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 152, 272], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f7df3e6c8e619e6ba7633d47e760efef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 4096], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_79f62956778921a8ec75eaabd31f2c76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7df3e6c8e619e6ba7633d47e760efef
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 4096], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7fd087e9b8d337f4e1b6470e74a6e682(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 4096], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_55657fb5204d24fac794f6677ae76fae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7fd087e9b8d337f4e1b6470e74a6e682
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4096], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_90b8be115cc76d7e1abbb94bb858c516(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 4096], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e8868d134dd8603196b1d6ebbcb5787b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90b8be115cc76d7e1abbb94bb858c516
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4096], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d801a3f63884f9364bf925bb56c70c27(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 196, 12, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_520d50ac545e6162b5f86a56660e559d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d801a3f63884f9364bf925bb56c70c27
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 12, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3c0688ceb8a47d57e27f99c90a19bb9b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 196, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0bc02d9fc19e5175109a2be4a3ac0f5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c0688ceb8a47d57e27f99c90a19bb9b
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5dff5785c503eefcb9c9b84253e71133(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 384, 49], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_543a9a27e8938df4d7737569184f8f3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dff5785c503eefcb9c9b84253e71133
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 49], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7f8042f2a9c21448974db77e26270663(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 2, 12, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a9061a2f676760b282a661fc485a3a55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f8042f2a9c21448974db77e26270663
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 2, 12, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_12174faa73e86c03464ec8a20a68da9e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 12, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2c67e2b8633c9d2c705cfd02392041d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12174faa73e86c03464ec8a20a68da9e
    def get_inputs(self):
        return [
            paddle.uniform([43, 12, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d498ad9776827c0c986ae19806e04706(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[128, 16, 8, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5f54deb3de69e2546fa67a198c85f642(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d498ad9776827c0c986ae19806e04706
    def get_inputs(self):
        return [
            paddle.uniform([128, 16, 8, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0caeab66e19b3a6b3f7f39b19662bc6d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[128, 320, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3b36a34356466b3b08c40ba4f397bbe1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0caeab66e19b3a6b3f7f39b19662bc6d
    def get_inputs(self):
        return [
            paddle.uniform([128, 320, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e619824f41249824280889dd4e6db702(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 15, 4096], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7521ee2398b8703704cdeb8d405b74a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e619824f41249824280889dd4e6db702
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_55657fb5204d24fac794f6677ae76fae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7fd087e9b8d337f4e1b6470e74a6e682
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4096], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4a65e69570ac7936e6fe370c04384258(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 91, 4096], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_10df1546204e938efab6a68c766e4a9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a65e69570ac7936e6fe370c04384258
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 4096], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_343f7d8725b39b02a9cffe0ced47c44c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 676], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_15a821c2e9a3ce80a5f70e762c6b6a27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_343f7d8725b39b02a9cffe0ced47c44c
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 676], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_870832a35002064bd31e1389c6b90bd9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 76, 676], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eade6aa72f266700c6e8dd7011827eae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_870832a35002064bd31e1389c6b90bd9
    def get_inputs(self):
        return [
            paddle.uniform([1, 76, 676], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e2d8367796f214c2af1f3a1adde0ed54(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 21, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_49512868da43d6bedb3b9f57db569b6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2d8367796f214c2af1f3a1adde0ed54
    def get_inputs(self):
        return [
            paddle.uniform([1, 21, 256], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d8090b6a7507ca030d8a62153e74211b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 120, 216], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2a4af43d5852570ebfd8c4c3adf32416(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8090b6a7507ca030d8a62153e74211b
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 120, 216], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a9b0d1c49431bb7c6bbbbf51b31485b7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 4, 5, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 8, 8, 128, 13, 13], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e274b2e3c59d2dad15cf7b1827e43f6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9b0d1c49431bb7c6bbbbf51b31485b7
    def get_inputs(self):
        return [
            paddle.uniform([4, 8, 8, 128, 13, 13], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_dc1e24823ee25b0d1804e4cd8855be49(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 900], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8b40b243b0e994e9220e10bdb6d11383(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc1e24823ee25b0d1804e4cd8855be49
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 900], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_30348972b680af2d50ed51e5652735c5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 900], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_74d96f9a327da4d82a00296272937ab4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30348972b680af2d50ed51e5652735c5
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 900], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_513a7aa4a27595b824e8e952e1bb0c0b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 2704], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f961ddc0424644abe0ad67633a6d981c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_513a7aa4a27595b824e8e952e1bb0c0b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 2704], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e25ce0e126b680a08732401d6e34c41d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 2704], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_123e342871eb23e593f4d14ff86ffafa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e25ce0e126b680a08732401d6e34c41d
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 2704], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a258dd145531ea97190a8cca89565af3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 3136, 3, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6675709c281afdacaff993e1129ec054(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a258dd145531ea97190a8cca89565af3
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 3, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cf4a2b21b5d3cf8f17e37d73fa57ab96(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 3136, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fdf1de3b8bec711d5ea1da6362107f15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf4a2b21b5d3cf8f17e37d73fa57ab96
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_031d36a3a5a0ece5a88d8d20d0a5c90c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 96, 49], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f26c8c429fb159e79864e2853b5d44cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_031d36a3a5a0ece5a88d8d20d0a5c90c
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 49], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8a1994d3cc6f890cb935370af05a2ce0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 2, 3, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_580bbc83bdf0af954669a9b3bf6833f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a1994d3cc6f890cb935370af05a2ce0
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 2, 3, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b1620b559ab272f1355a37d28ad3736f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 3, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_39574aef81d90958f0d63e7a55dec2b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1620b559ab272f1355a37d28ad3736f
    def get_inputs(self):
        return [
            paddle.uniform([43, 3, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_595f21ae57133de93b9d170c3981407a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 320, 3, 4, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0ad02fa41dca084fca23a2efa148a5ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_595f21ae57133de93b9d170c3981407a
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 3, 4, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_768c972b03a389502e72873a3e86a2fc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4, 320, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c11048827d52206d6b94ca26cce54cef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_768c972b03a389502e72873a3e86a2fc
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ccf813327c1df683ffd36b417520bc5d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4, 320, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_601f6fe9d2c4f02398643696a25312ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccf813327c1df683ffd36b417520bc5d
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1e2570e41060c51200cb421154d7b92e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 15, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9fdb05006dd0367e15040fa77a5dc610(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e2570e41060c51200cb421154d7b92e
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4f6c6361dc5654339d27ac533307ae9c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 169], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0dfc4dd5695b435a5c4fc23fc1e17f30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f6c6361dc5654339d27ac533307ae9c
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 169], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1214c097f1a2d9e402428cebe068787b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 76, 169], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9670ce9e686f967d708373ee927d75a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1214c097f1a2d9e402428cebe068787b
    def get_inputs(self):
        return [
            paddle.uniform([1, 76, 169], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0c99416cbf0cabc11d5a9f6b62091af8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 32768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_55d9ebe1ddd09412f83d03cf1fe0ef84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c99416cbf0cabc11d5a9f6b62091af8
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 32768], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_90ad315cd08e56baec1c7e0b247f5a9f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 19, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ca4bb42210bf76fb4674f150b1aad1f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90ad315cd08e56baec1c7e0b247f5a9f
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9874f711042e2bfd107011a52d643e65(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 15, 16384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9e731d2e3c529ff0307c78684bca1591(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9874f711042e2bfd107011a52d643e65
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 16384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_736429e2d6de83812117a67bd8bf0eec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 16384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aa5fab6332e1433b0403dc03e2d1b60c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_736429e2d6de83812117a67bd8bf0eec
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 16384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1c9267bec5ee55f9afae9c6d978b6d09(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 91, 16384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_169bc8da91ad15cf7be39a93e325cde8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c9267bec5ee55f9afae9c6d978b6d09
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 16384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e4542efefe8f666234f64f3301bc1ef7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 1156], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a1356c9f2341d1f00a3f12c305c2770a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4542efefe8f666234f64f3301bc1ef7
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1156], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_75e983b492c1427fd1695e8b1cd8c424(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 1156], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c7ea7c8025891d1399b86a36e3f90cdd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75e983b492c1427fd1695e8b1cd8c424
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 1156], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_df87dbc8e265819adc369d8222e2dc5b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 7581, 4, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fc1e08dcc71f1bb085a06af24d09ca0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df87dbc8e265819adc369d8222e2dc5b
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_06ffe708a0cca10cc183bf23d601a5e0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[528, 4, 96, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fa27f311ae348d00f1a70914a410bd17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06ffe708a0cca10cc183bf23d601a5e0
    def get_inputs(self):
        return [
            paddle.uniform([528, 4, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_62fb83302cfa4bef7f18171cd80aa468(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 1, 24, 48, 2, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_45f2acfb96d6674c7ec4eef35d36c65b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62fb83302cfa4bef7f18171cd80aa468
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 24, 48, 2, 96], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_007f0eb0cc5b8014b740db1d961f6253(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 5776], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_72301a82b76119d0ec2f0bfa0e14191c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_007f0eb0cc5b8014b740db1d961f6253
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 5776], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c00cd60caf07dddd01fb3df6bd2469b8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 5776], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4e46b67ecc9766d589a4b5e93454e63d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c00cd60caf07dddd01fb3df6bd2469b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 5776], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bcb65fe6d53f3d62ff76d7cbac56b040(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 4, 7, 4, 7, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e30b7c094389d9f78baba55030edf63f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb65fe6d53f3d62ff76d7cbac56b040
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 7, 4, 7, 192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_329d5670637f8070db71c773aa95c706(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[3, 0, 1, 4, 2, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 16, 49, 3, 6, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0f336e1968bad2e272ce1799ddcab1dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_329d5670637f8070db71c773aa95c706
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 49, 3, 6, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3412bda6865389dff7f108ee1309863f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 2, 4, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 16, 6, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e909c523570bc891756685db3c798f1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3412bda6865389dff7f108ee1309863f
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e7c031aaaac960f4b4a468a979139505(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 16384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1a2a20942cb6d23d584e10b115094655(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7c031aaaac960f4b4a468a979139505
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 16384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3a12c3fdb2e6599927733e10ca561501(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 21, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e437089afa09d95e83bc9dea1e3b5c5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a12c3fdb2e6599927733e10ca561501
    def get_inputs(self):
        return [
            paddle.uniform([1, 21, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_470489840a6126a06e53c706c1a1146f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_581351d2c9f222d4e806d2f588dfd829
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 576], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_49d1af5fc0f630b1908640a479a3a8f4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 576], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c446e16f8b90b352fb0978e94529cb2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49d1af5fc0f630b1908640a479a3a8f4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 576], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d10bbc13eac969e4ab1bf7e029eba0c0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 576], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2e5e355f30104036d06f0640f5cb72f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d10bbc13eac969e4ab1bf7e029eba0c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 576], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a18a2355f862818d6ad8b6fe7b9636dc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 2, 1, 12, 24, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1776e4e9ab9e6b54cb5b5b004eed363c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a18a2355f862818d6ad8b6fe7b9636dc
    def get_inputs(self):
        return [
            paddle.uniform([6, 2, 1, 12, 24, 192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b48eebce2ab72775fbd82e19d4ee3621(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8, 16, 64, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c83c3e6634c487f369d99fe692df17e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b48eebce2ab72775fbd82e19d4ee3621
    def get_inputs(self):
        return [
            paddle.uniform([8, 16, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b2aac0f78dcddd109cb8aba8bf377251(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8, 320, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fe8fd9a95469a9ae159a9a170e981ab5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2aac0f78dcddd109cb8aba8bf377251
    def get_inputs(self):
        return [
            paddle.uniform([8, 320, 256], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6c0289ebf5e4ed59e194ad32b4dc2244(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4725, 4, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_80db9773d50af597e9eb3a3070265d7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c0289ebf5e4ed59e194ad32b4dc2244
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_547c93acd957ac419387010880e0d76f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8, 16, 64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f25ca28a6e351bf19eb201f8c9504917(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_547c93acd957ac419387010880e0d76f
    def get_inputs(self):
        return [
            paddle.uniform([8, 16, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1523539423b68b167a7a740688f5e356(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8, 160, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4981742328f13503c92119c5399dc9c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1523539423b68b167a7a740688f5e356
    def get_inputs(self):
        return [
            paddle.uniform([8, 160, 256], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_64f1c49b4c4b7a7267ce1e66799e24c6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 12, 12], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_43f0f32dd250d7b6af4b99db6b24cd3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64f1c49b4c4b7a7267ce1e66799e24c6
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cda9d6cf5a776dc418507f5a12dca5cf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 577, 3, 12, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9e226c89938a8253a18525d43295938c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cda9d6cf5a776dc418507f5a12dca5cf
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 3, 12, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_92074a94b7fdbc7fee4e192de5aefa56(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 577, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8bc04acc61a48872f664f910c5552cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92074a94b7fdbc7fee4e192de5aefa56
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 577, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_471ed70a60e333d36570a03c211f3c51(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 577, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7e882cf2b087cb4181daba02a949d432(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_471ed70a60e333d36570a03c211f3c51
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 577, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1b6c8293f65831c9334a0fc0cb0ee6f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 1296], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f023174d1510972d0dedca1f7cc03702(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b6c8293f65831c9334a0fc0cb0ee6f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1296], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6484eab9bbe32febc07cad4cae37bb83(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 1296], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_437f42e31fb24d4c153091a5afb228a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6484eab9bbe32febc07cad4cae37bb83
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1296], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e3ffebd65568785065b6f509cc63465f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 1296], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e35c536e824f6dd8862ef16e0297562e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3ffebd65568785065b6f509cc63465f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 1296], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a6e9cccd541b405aab72c9acb51477eb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[64, 64, 16, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_634655d1d62329c7492fb3269c693a03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6e9cccd541b405aab72c9acb51477eb
    def get_inputs(self):
        return [
            paddle.uniform([64, 64, 16, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5ff125aa61ab73a811912ccf51c245da(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[64, 64, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_87f735479262cdc895370299d4c402f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ff125aa61ab73a811912ccf51c245da
    def get_inputs(self):
        return [
            paddle.uniform([64, 64, 256], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e08a65ff36de22a726435f401f84d918(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 197, 2, 6, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6081e68de45ab347052c8aaea339ebfe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e08a65ff36de22a726435f401f84d918
    def get_inputs(self):
        return [
            paddle.uniform([10, 197, 2, 6, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_26b4b23e8eb87b507557c1abf9d15315(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 197, 6, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ab311b8d38670efed0b7aa057f153da5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26b4b23e8eb87b507557c1abf9d15315
    def get_inputs(self):
        return [
            paddle.uniform([10, 197, 6, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d8df2b47cde8af75fad934f25b03fbb5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 6, 197, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6fb681b067e84d5acf87cfa0837fbb62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8df2b47cde8af75fad934f25b03fbb5
    def get_inputs(self):
        return [
            paddle.uniform([10, 6, 197, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f735e5cf39f651171a1eeb829d84cd1e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 15, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f3eaf110e733e1263e2df8bb462180b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f735e5cf39f651171a1eeb829d84cd1e
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 256], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3a6321911f0ab1f806208e5a85cef5b8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[384, 2, 96, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3f686877f3494a5dd0fa45c9c79a3093(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a6321911f0ab1f806208e5a85cef5b8
    def get_inputs(self):
        return [
            paddle.uniform([384, 2, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_66ba65cd188fb55fcc87ad08b171a477(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 1, 96, 96, 1, 48], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_93da0c3181e442381f092df36368e121(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66ba65cd188fb55fcc87ad08b171a477
    def get_inputs(self):
        return [
            paddle.uniform([4, 1, 96, 96, 1, 48], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fb85ac39ae2f603bb1d17b3050053cd5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 4, 7, 4, 7, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7bd0d7befef9ab7751dfd29b0f1f2853(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb85ac39ae2f603bb1d17b3050053cd5
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 7, 4, 7, 192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4e09661471d5d54e80c75cb6b070a2ed(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[3, 0, 1, 4, 2, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 16, 49, 3, 6, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2614844d310128078671981c398d53d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e09661471d5d54e80c75cb6b070a2ed
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 49, 3, 6, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_dad816d24251c1ac6c20aecd26b8083a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 2, 4, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 16, 6, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0e675acc5ce45e64abff54a2d7d27f17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dad816d24251c1ac6c20aecd26b8083a
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_061990513692303df41a98fcfcdad53b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 16384, 2, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_18db64e249e72d83e8dff3488973dcd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_061990513692303df41a98fcfcdad53b
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 2, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_73bf8332ea320a33303e7f5775e6fb38(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 16384, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_089c686ac36f9db7a7e3c576160aa8cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73bf8332ea320a33303e7f5775e6fb38
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_88755be9843ba16dd335ef62bd59d0d4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_20d6a6801e611a4df82879fedd95f028(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88755be9843ba16dd335ef62bd59d0d4
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_404571ecb47f2fd6283cc822edbea92a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 2, 2, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d029d459a9afdc051b39f358119d1e95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_404571ecb47f2fd6283cc822edbea92a
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 2, 2, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_68d3c84b330865985ed3821f719c5c1c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 1024, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_75a8645f9633bd34911bf4b639420e3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68d3c84b330865985ed3821f719c5c1c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 1024, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f716c4c65796fad43a80df44c390c376(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 16384, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_00fadb7e2212ecb4e4f01224a55cc231(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f716c4c65796fad43a80df44c390c376
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16384, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7521ee2398b8703704cdeb8d405b74a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e619824f41249824280889dd4e6db702
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6675709c281afdacaff993e1129ec054(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a258dd145531ea97190a8cca89565af3
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 3, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fdf1de3b8bec711d5ea1da6362107f15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf4a2b21b5d3cf8f17e37d73fa57ab96
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f26c8c429fb159e79864e2853b5d44cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_031d36a3a5a0ece5a88d8d20d0a5c90c
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 49], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_580bbc83bdf0af954669a9b3bf6833f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a1994d3cc6f890cb935370af05a2ce0
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 2, 3, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_39574aef81d90958f0d63e7a55dec2b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1620b559ab272f1355a37d28ad3736f
    def get_inputs(self):
        return [
            paddle.uniform([43, 3, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a5d19a2f17bfdaebd8fddebf17a5f21f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16, 32, 64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_972262a6b2e534ee71f134f711bfe291(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5d19a2f17bfdaebd8fddebf17a5f21f
    def get_inputs(self):
        return [
            paddle.uniform([16, 32, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_54924395bf823834af595d02c1bf54cd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16, 128, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8c388a868904188b4c9e6e49cacb0a38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54924395bf823834af595d02c1bf54cd
    def get_inputs(self):
        return [
            paddle.uniform([16, 128, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0dfc4dd5695b435a5c4fc23fc1e17f30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f6c6361dc5654339d27ac533307ae9c
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 169], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_398e3e4874eca6a3306119ffa4329d14(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 169], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e0fcff71aeb00a2fdd8f9b0e06170db0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_398e3e4874eca6a3306119ffa4329d14
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 169], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c69235fd3d31c63159f07f5ad042bf71(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 24, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ce6bc6ee59e259d9d2538933c20cfee3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c69235fd3d31c63159f07f5ad042bf71
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 24, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5fb31d094233c7b13fc7ef772520913e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 2, 24, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bab23deca8c0edfcf5f56d5b9e45f7d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fb31d094233c7b13fc7ef772520913e
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 2, 24, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c85fcaddcc667107ada757cc01b54fae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 24, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5015c130c56d150aaecba54613056e5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c85fcaddcc667107ada757cc01b54fae
    def get_inputs(self):
        return [
            paddle.uniform([43, 24, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ca161ce2677a5dd4733c58d95a2ca851(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 8, 8], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d16ed86fc8d35930976e8233331452b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca161ce2677a5dd4733c58d95a2ca851
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_710e3c50e431fd73026438794cd93038(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8400, 4, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3667648a9799726649af8dd4831ac947(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_710e3c50e431fd73026438794cd93038
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4dad0e76957d256859563cd800db89cc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 400], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_49c0ee6a1a57cc8338757edd3c26b2ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4dad0e76957d256859563cd800db89cc
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 400], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6c51269703391cb3c9db022ed4a74d00(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 400], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_75abc81b00c0d0b962b6387f7e95f4b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c51269703391cb3c9db022ed4a74d00
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 400], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_450cea39ff2ac547a5519433d91a0634(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8, 32, 64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e554008639bbf0f3dbb2ead1a2684b41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_450cea39ff2ac547a5519433d91a0634
    def get_inputs(self):
        return [
            paddle.uniform([8, 32, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6e21ce9160cb9e4b4ea718d5dbbc1ca6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8, 160, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_521a2d450984345e36d5429dd8881065(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e21ce9160cb9e4b4ea718d5dbbc1ca6
    def get_inputs(self):
        return [
            paddle.uniform([8, 160, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ef34883d20f522e17d7e43049b0030f4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5a424ceed48a127613f4f0661535f605(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef34883d20f522e17d7e43049b0030f4
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4f4acf3d5f02491adbefa653fd95f255(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549, 4, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8e27279f65705843429e9eafb78c7e4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f4acf3d5f02491adbefa653fd95f255
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d77a2a27951037085a3b255bd345b385(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 768, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_830d88adf140eb9080dd338a98291e3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d77a2a27951037085a3b255bd345b385
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c489af8772de81a0cb76656183cef816(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 60800], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d8926ae41c21c507b639fa679936c3b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c489af8772de81a0cb76656183cef816
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 60800], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b399c67bb568f1d6cc4b530cab89b374(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 60800, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_658ca37606ccc3cf74d16fc7f973d125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b399c67bb568f1d6cc4b530cab89b374
    def get_inputs(self):
        return [
            paddle.uniform([1, 60800, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d8926ae41c21c507b639fa679936c3b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c489af8772de81a0cb76656183cef816
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 60800], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_dfbe3a6aa43880fbf6cebecb511fe828(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 640, 3, 2, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5bea1a4623d6e7d81296f0fa8c22dd9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dfbe3a6aa43880fbf6cebecb511fe828
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 3, 2, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_621f81f1cbecca85acf62772f42769cf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2, 640, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dd5f2f98ae9f3ce672a0933c06c639e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_621f81f1cbecca85acf62772f42769cf
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 640, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6f363264539178be5ec5a91ba6de5e6b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2, 640, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1f9018e8087988d2523c613fe5bc5346(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f363264539178be5ec5a91ba6de5e6b
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 640, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_60ea5d8c4ace25f3e000326b318f7328(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_24bffe33065e6883f6c26c7142e6982a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60ea5d8c4ace25f3e000326b318f7328
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7c1e1416d893818f1de03187f7eedcd6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 198, 3, 3, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6be583e3bac79dbc3b7f851ad61bcb24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c1e1416d893818f1de03187f7eedcd6
    def get_inputs(self):
        return [
            paddle.uniform([86, 198, 3, 3, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cb568aa5b4189bedd9b0f56484dd1b3c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 3, 198, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8661ccacbb77e6a4c9bd1f71eb172a52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb568aa5b4189bedd9b0f56484dd1b3c
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 198, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_973be8ca93bb921497b74593c93cb1d1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 3, 198, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_700a25fc0ce1463f76ed580f3c73fadf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_973be8ca93bb921497b74593c93cb1d1
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 198, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0c1527eec68163910f412fc52da11d63(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 1600], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b2c5397ab697908d4bcd7cc452f9f279(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c1527eec68163910f412fc52da11d63
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1600], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_dc4e16bdca22ccda1ee9297bb7762fd5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 1600], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_48b02826f3fb3b3e95bc50c86413d041(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc4e16bdca22ccda1ee9297bb7762fd5
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 1600], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ded1a7c05af155087ddb75b3ab878e1d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 3136, 3, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_400673a6d7a75cf3cf46e6fe57fb5fe8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ded1a7c05af155087ddb75b3ab878e1d
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 3, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_861b55d23659bfb7e30bb07acda5fed3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 3136, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ef692944fddf2b13c0706903f9030fe6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_861b55d23659bfb7e30bb07acda5fed3
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b1c5ea8fe5310ea4b3cb7a0cda27ec5d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 96, 49], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_da4ce96ab6c131153d81411b696a1565(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1c5ea8fe5310ea4b3cb7a0cda27ec5d
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 49], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_db74162f4e6a34eb35457836172edf90(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 2, 3, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0164606818aab1c09d695d6cd0c8afd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db74162f4e6a34eb35457836172edf90
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 2, 3, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1c21259e70f761ede775e12e07f6f854(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 3, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d67028909bf9abf7ce21096e1404246c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21259e70f761ede775e12e07f6f854
    def get_inputs(self):
        return [
            paddle.uniform([11, 3, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f8285ec42634319f2e049a5aef4128fb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6ad2910b014ab5563a0e162d0bdf1516(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8285ec42634319f2e049a5aef4128fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 256], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_89bd00f141b3a0ecd5dc3412d8fc13dc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20, 8, 288, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6ee09beb196bd7344d6e6f875932488e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89bd00f141b3a0ecd5dc3412d8fc13dc
    def get_inputs(self):
        return [
            paddle.uniform([20, 8, 288, 24], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e30c24680d5b1c82f855eb5775810692(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 1, 2, 24, 12, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_67c90834526afc05fcdcf153ac7f68e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30c24680d5b1c82f855eb5775810692
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 2, 24, 12, 192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_80363010a1f116d3982d083987d5a0e5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 196], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7b9475c5ca01f4205301632f197af5ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80363010a1f116d3982d083987d5a0e5
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 196], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_31adf99fcd3dc5d1ee65c307e60599d5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 196], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_93031fab65bd24810bb41513359fbcc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31adf99fcd3dc5d1ee65c307e60599d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 196], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1247e5974bcf831b85bff78035225f53(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 1, 7, 1, 7, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_93cbe1276b633946cd197eaccf209986(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1247e5974bcf831b85bff78035225f53
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 7, 1, 7, 768], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_92dedaec40aad3f7c347bb59c85ff802(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[3, 0, 1, 4, 2, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 1, 49, 3, 24, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1dd6deb8b50611947da83583a90339b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92dedaec40aad3f7c347bb59c85ff802
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 49, 3, 24, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f5956615907214bfca4df75c5c979ad5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 2, 4, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 1, 24, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4dd241240974bc91778db703596d2bbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5956615907214bfca4df75c5c979ad5
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4e2f5bea49ff89ad332d2a3e44432b5f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4312, 16, 2, 4, 6], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dda617dd67c30936995cf36e2711a192(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e2f5bea49ff89ad332d2a3e44432b5f
    def get_inputs(self):
        return [
            paddle.uniform([4312, 16, 2, 4, 6], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_134f860ea1405df8ba3d6cf28c13342a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4312, 16, 4, 6], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a1d4e6945547687f9bf5e54470808009(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134f860ea1405df8ba3d6cf28c13342a
    def get_inputs(self):
        return [
            paddle.uniform([4312, 16, 4, 6], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_24df84a92a70abf73ccdca04b0d3bfe9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4312, 4, 16, 6], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f2fff269efb099d27a31598eebdacc6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24df84a92a70abf73ccdca04b0d3bfe9
    def get_inputs(self):
        return [
            paddle.uniform([4312, 4, 16, 6], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_645fb0c2c28c847a8530df8cb42db822(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_112b2548816103019986f69677d7b728
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 2304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2679ef5e7b61beb1ce38752d40d6a686(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_291f7e7c973ea004a299570d6cd22ca0
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 2304], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d8d05272f39c9467c5f3664f33ed070e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 441], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1b2c81da40674ac1598bdab7eb818c77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8d05272f39c9467c5f3664f33ed070e
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 441], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d65f6f30ff0da12ab56538d3eb16821c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 441], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e8a8b46f53ab0f98181de5fa229a0219(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d65f6f30ff0da12ab56538d3eb16821c
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 441], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_343e76464d282b75911f1e27a5b17fdf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4be89125fd8bfec5dde5279597f614
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 7, 8, 7, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a1ef02ae7cb80742dda80c29c9e03484(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c09db675753c27b512b2ff6d36a883ef
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 49, 3, 3, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_39baf3381351fc67df513864b7d8ae3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a821b35effdac8f38776368a78c6e04e
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a1356c9f2341d1f00a3f12c305c2770a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4542efefe8f666234f64f3301bc1ef7
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1156], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c7a2ba69166fa9116f9176e8fdebfb74(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 1156], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d3314aec2659cd9b2b041451a34af731(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7a2ba69166fa9116f9176e8fdebfb74
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1156], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e6bc6d6b53ad9041b8996539f44e9f0b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 1156], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_52f79bab0c051945ed1498b3819c7ff3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6bc6d6b53ad9041b8996539f44e9f0b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 1156], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_edaf98c27f2d766d47e342429cff57ab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4096, 1280], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_68060c659ff9b699b832acfcfe3bb9f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edaf98c27f2d766d47e342429cff57ab
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 1280], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_17a8169257d5015aa28d9d10885b817a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1280, 4096], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c7a763c215602a21da1f8456f227a027(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17a8169257d5015aa28d9d10885b817a
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 4096], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_48a4ff4883b2b1c63f9403eb1f027eba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 176, 264], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4cb8c42b1f5e23c90aaafb60b7fc266d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48a4ff4883b2b1c63f9403eb1f027eba
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 176, 264], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_db10f8c5ecad126da2b59f0af899623c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 88, 132], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_734bdd27474d456855bb2c9b0307d3f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db10f8c5ecad126da2b59f0af899623c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 88, 132], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_915e098e1d32d8d935f3b03b0f1535c0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 44, 66], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7ea532a292bbe3c4d743a606ad2cd582(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_915e098e1d32d8d935f3b03b0f1535c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 66], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6ee36dff715dbbb15f0943d5f1e0b8ad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 22, 33], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_65084e7185d3a7656e33f973c2ccb0f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ee36dff715dbbb15f0943d5f1e0b8ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 33], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_deef41e3f816968baebc657eb4759f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f8f98bda767ec9a08d1da019e7a1692
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b11306caa279b647442d9aa76c4636b2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 176, 264], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7b41ada1e7f77928a608a5ebd3867ff0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b11306caa279b647442d9aa76c4636b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 176, 264], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fd86645fd2e06be3a4690944602ce701(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 88, 132], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_39a8212ccb03e6c24d717ec33f145062(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd86645fd2e06be3a4690944602ce701
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 88, 132], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_89a655b83039e313be97e4657a80d300(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 44, 66], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d54ad24cff2b5235b95bb7bfdd826079(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89a655b83039e313be97e4657a80d300
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 44, 66], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_75508c87eafbc3169991cd511bf194a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 22, 33], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_74363fd005d51d3d0a0abdfc6c67c498(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75508c87eafbc3169991cd511bf194a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 22, 33], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d97a2715657d571c415936c2987dead(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1144ac8a6473d933300e925b1b922cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 11, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_80be6c6f32872568cb8aafb48c951313(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32, 65536], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bbaeda9a63f9274ae2e45971fa0ccf30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80be6c6f32872568cb8aafb48c951313
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 65536], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_91837bfe3f4c5b6ca4739092fcd3eb11(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[576, 2, 96, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b00e6eda821b6c0a989615480f55a766(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91837bfe3f4c5b6ca4739092fcd3eb11
    def get_inputs(self):
        return [
            paddle.uniform([576, 2, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_27a475bfa30153be56fa2378ea51d80e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 96, 1, 1, 96, 48], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cf3092532c69f9f867a1699e78106849(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27a475bfa30153be56fa2378ea51d80e
    def get_inputs(self):
        return [
            paddle.uniform([6, 96, 1, 1, 96, 48], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8f6cc4f8538fe217a0a20e32b2614dcc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 324], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d0fe5b126607faf492f8fdcf24da9971(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f6cc4f8538fe217a0a20e32b2614dcc
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 324], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fcff9d7ee34b55bd04d3aeb43a9e4fdb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 324], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_41017bc189be78fae8f5deaece43b767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fcff9d7ee34b55bd04d3aeb43a9e4fdb
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 324], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8a9d0e2c307efff9f565707ba5afb8bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 324], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b5fd6dec09b4f92d504c562983f3554c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a9d0e2c307efff9f565707ba5afb8bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 324], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f8c309925bd041e71bab9a02d33bbd85(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 19, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ca7631af8e5aede2d354c8ee104c775f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8c309925bd041e71bab9a02d33bbd85
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 256], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_82aa51e07a4573708aef60fcd33c2833(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 289], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d2f94aee6e32070317449d89b1d62221(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_82aa51e07a4573708aef60fcd33c2833
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 289], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d778c7e6041fc4f84759767a43fb225e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 289], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f2678c8a91cf5ccaa05bad83ce7c217e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d778c7e6041fc4f84759767a43fb225e
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 289], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_47e9518fa8d3423f1a4eb32c1d5076ea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 289], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a5616b05e0b2acf9aead90138e307c13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47e9518fa8d3423f1a4eb32c1d5076ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 289], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0ef7f11bc4e7114e279e1f498e29fdd6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 96, 9216], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c7483b1fa860d2399b136bfad0059aba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ef7f11bc4e7114e279e1f498e29fdd6
    def get_inputs(self):
        return [
            paddle.uniform([6, 96, 9216], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0146e318a3ec2a9a2a35abef5c497578(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 32, 144, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_73a0c7b53ed6b4c159a62fd644893c79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0146e318a3ec2a9a2a35abef5c497578
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 144, 24], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c35646adc6a905cabf4ac542a1f4d40d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 1, 1, 12, 12, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_22a51de842773f757248e79eb634cb95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c35646adc6a905cabf4ac542a1f4d40d
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 1, 12, 12, 768], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_15155b54080fdcbb2d74186114d247f5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[96, 4, 96, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7f43a15f0e1b9b6d5e46ba337dc095c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15155b54080fdcbb2d74186114d247f5
    def get_inputs(self):
        return [
            paddle.uniform([96, 4, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_22b42e72eedbc9f3e976655e247026f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 1, 24, 48, 2, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8b95194094b96aa3adcefb0ca5fbebfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22b42e72eedbc9f3e976655e247026f9
    def get_inputs(self):
        return [
            paddle.uniform([4, 1, 24, 48, 2, 96], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_92bde32ef0726228edc481e9b579de67(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[12, 8, 288, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fcda12cbbd17147bc26608ba92687ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92bde32ef0726228edc481e9b579de67
    def get_inputs(self):
        return [
            paddle.uniform([12, 8, 288, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1776e4e9ab9e6b54cb5b5b004eed363c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a18a2355f862818d6ad8b6fe7b9636dc
    def get_inputs(self):
        return [
            paddle.uniform([6, 2, 1, 12, 24, 192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ea094c91674020391a25acb9352d7b6f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 8, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_73457a5d5b892188d06250f258d1187c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea094c91674020391a25acb9352d7b6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 8, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f31cc463b3c7cfaf662022ee75374305(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 2, 8, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fba09d1bfa393d0ec14e6894845e1f68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f31cc463b3c7cfaf662022ee75374305
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 2, 8, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7403db6d7a89fa0d9a7d07bf52e5d599(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 512, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8261744c606435094f4591fe8188ad1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7403db6d7a89fa0d9a7d07bf52e5d599
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bff2ac97d1f50cd3c371e50f9a590d0d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 512, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_929babc8372e04f5ad286bf31235e0dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bff2ac97d1f50cd3c371e50f9a590d0d
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f4a3f80c348ab2536c1285fdd1064a90(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 3136], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cccc03e71ccab4edf756b2cca77912fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a3f80c348ab2536c1285fdd1064a90
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 3136], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_93fff2ee76cb12b87038ad024d7858eb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 3136], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_542219d27cf4f2511c63e867c4eb6efb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93fff2ee76cb12b87038ad024d7858eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 3136], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c1320b33acaa23de10bfcbfb7345f998(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 32, 144, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6fa6dd1c4208d84af90f89776c2caa73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1320b33acaa23de10bfcbfb7345f998
    def get_inputs(self):
        return [
            paddle.uniform([6, 32, 144, 24], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_78745594ce305264858c3f517cf6f24e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 1, 1, 12, 12, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_02e257a6d71ad94adbc6aa02a500e2ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78745594ce305264858c3f517cf6f24e
    def get_inputs(self):
        return [
            paddle.uniform([6, 1, 1, 12, 12, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7521ee2398b8703704cdeb8d405b74a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e619824f41249824280889dd4e6db702
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_55657fb5204d24fac794f6677ae76fae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7fd087e9b8d337f4e1b6470e74a6e682
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10df1546204e938efab6a68c766e4a9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a65e69570ac7936e6fe370c04384258
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 4096], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e9bb38c29a78dd6b9d08b9b432d8c815(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 20, 196], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fffe534306c986f30694303152f13e3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9bb38c29a78dd6b9d08b9b432d8c815
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 196], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_93031fab65bd24810bb41513359fbcc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31adf99fcd3dc5d1ee65c307e60599d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 196], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c8720a1a95dd8cd9358aef827dc3a5f6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 196, 12, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c16e23652ba9807698ac09d1a8404d04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8720a1a95dd8cd9358aef827dc3a5f6
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 12, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_50e179c887d95e472bab06e86b01d414(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 196, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0f288fbdc80ad394b57bad99af86f431(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50e179c887d95e472bab06e86b01d414
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fb9eb20474bb4e4f7d0f85180b7cb4a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 384, 49], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d3c5d441fc5f1833890152fc2d40f6a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb9eb20474bb4e4f7d0f85180b7cb4a6
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 49], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_07bfc4e5f5542767c9d6eaecc9872d75(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 2, 12, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_787e37ff7bda6da9deb2e327f11f5265(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07bfc4e5f5542767c9d6eaecc9872d75
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 2, 12, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8cbc16abd8a3736cc9be4a9e2478f813(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 12, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_69fdf6eae5e56d8d2ca856a3e250f29a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cbc16abd8a3736cc9be4a9e2478f813
    def get_inputs(self):
        return [
            paddle.uniform([11, 12, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a9a98da78b84aaa182a7264c6f891387(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[960, 2, 96, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7dc0fada843c115bf13ad5e721e27199(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9a98da78b84aaa182a7264c6f891387
    def get_inputs(self):
        return [
            paddle.uniform([960, 2, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_40bd8253c7a8271dd1fecc8f502ce64a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 96, 1, 1, 96, 48], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1ca75ea50a59bf1094c8f73b141a4b88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40bd8253c7a8271dd1fecc8f502ce64a
    def get_inputs(self):
        return [
            paddle.uniform([10, 96, 1, 1, 96, 48], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e2e2307ca0e823e177d317d8479f8a9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4590d695db9f8983f1bc26345670654
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 3, 4, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_00294c4b9aa581ba63722bb4d9a5c9df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c6da0f0264d7892c1de7fb8d258e236
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ec2a1073a4bf822e37ab93e0566c419c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_138bae9e0cfad9b1e553a4cf0b6144a3
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_25224b976a4c62f9b73c99177728f133(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 16, 38, 38], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_82ee50d7e67c30496ed69883e8966be5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25224b976a4c62f9b73c99177728f133
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 38, 38], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a7d0a429d7247dd133d432fd00638fad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 84, 38, 38], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a26f5208f40628a3c95fa535ed77cce7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7d0a429d7247dd133d432fd00638fad
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 38, 38], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b718b8fdbb7a99e8261977231a5b74b6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 24, 19, 19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a7fa39722a8907b8d37b5e433b4ce6d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b718b8fdbb7a99e8261977231a5b74b6
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_53349c9886a0c960a5920f0066ab9a74(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 126, 19, 19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eda4f214dce67c3abb408d636cdc7a82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53349c9886a0c960a5920f0066ab9a74
    def get_inputs(self):
        return [
            paddle.uniform([1, 126, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_90ed91c7719afdfa3eb9e41c06bb7e64(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 24, 10, 10], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3eeb405cfa319a8f4064f98a5635f9a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90ed91c7719afdfa3eb9e41c06bb7e64
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 10, 10], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ad71017a058003c28d0a82342c6e0fc4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 126, 10, 10], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a310a82337e73dd3b9c4e8705c0449fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad71017a058003c28d0a82342c6e0fc4
    def get_inputs(self):
        return [
            paddle.uniform([1, 126, 10, 10], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cbd6962fbf999360ca24f7d39466e6ec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 24, 5, 5], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_028429f9d122c3566f40f13d577afbed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbd6962fbf999360ca24f7d39466e6ec
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 5, 5], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6fec08892a6349a5f468a4207e7cd26e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 126, 5, 5], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_41651d1802f94da15a192fa87df37a58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fec08892a6349a5f468a4207e7cd26e
    def get_inputs(self):
        return [
            paddle.uniform([1, 126, 5, 5], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_739eb0dbf3d157771d0082621e573fe8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 16, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_61cac30378bb6ebcb11fd2b4911c0d88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_739eb0dbf3d157771d0082621e573fe8
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7d43255a16ed6a9bfdf93993eb6a5fda(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 84, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_00e407eec634f3bc1fad564c9eb7f62a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d43255a16ed6a9bfdf93993eb6a5fda
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f3e4371bb12f62f093ca646b7900bae8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8f19b08fc9ef64385ba379dd09632445(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3e4371bb12f62f093ca646b7900bae8
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[16.527170181274414]], [[15.202004432678223]], [[17.437217712402344]], [[17.083877563476562]], [[16.537782669067383]], [[17.370038986206055]], [[16.433361053466797]], [[16.837352752685547]], [[17.173891067504883]], [[16.521339416503906]], [[15.94709587097168]], [[19.124284744262695]], [[18.33331298828125]], [[16.955554962158203]], [[16.09485626220703]], [[16.738065719604492]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class PrimitiveOp_1ab02cff6dfda5a85b26a6b1aaa0c9a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 84, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7fa6f7c7adf20ee18b9ed7f479b77c7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ab02cff6dfda5a85b26a6b1aaa0c9a6
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f56584f9f9e80530fc61f4e2165312c1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2112, 2, 96, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_295809ac85d2183b72c14b29e67a2f31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f56584f9f9e80530fc61f4e2165312c1
    def get_inputs(self):
        return [
            paddle.uniform([2112, 2, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f1472c411c7d21a4c2b4b8bda078a569(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 96, 1, 1, 96, 48], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4d2dbe27d8dfe64e1b7e21cde71261da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1472c411c7d21a4c2b4b8bda078a569
    def get_inputs(self):
        return [
            paddle.uniform([22, 96, 1, 1, 96, 48], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_07c30113b3c1cdb0fdb34058a45dba4f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 36, 28, 50], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ef7b9b769a61583186bf9af0fc8025f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07c30113b3c1cdb0fdb34058a45dba4f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 36, 28, 50], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f20086c37f41b11be18424d31e4e562f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4116, 4, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7d4262461a8c3a00e78eb23e2e12c200(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f20086c37f41b11be18424d31e4e562f
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d2f94aee6e32070317449d89b1d62221(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_82aa51e07a4573708aef60fcd33c2833
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 289], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a2c102b3ccf6d6de8521fd7d4ef7a611(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 289], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_af256acd01988ab995a82c58689b2c44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c102b3ccf6d6de8521fd7d4ef7a611
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 289], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5873366485593aa1e6a050d8ea9217b3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 49, 8, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_993b6f91c7c33514cbd5994bb0bdf580(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5873366485593aa1e6a050d8ea9217b3
    def get_inputs(self):
        return [
            paddle.uniform([22, 49, 8, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_993b6f91c7c33514cbd5994bb0bdf580(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5873366485593aa1e6a050d8ea9217b3
    def get_inputs(self):
        return [
            paddle.uniform([22, 49, 8, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1bb08b3d24e669aaf94be7af5543c94b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 49, 8, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_87fe4c3489d9faef328b22fb5d6031c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bb08b3d24e669aaf94be7af5543c94b
    def get_inputs(self):
        return [
            paddle.uniform([22, 49, 8, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8349f2a49b9231f7c4e61632f7b22ed4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 8, 49, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_26e0caff3718b3e85cc70a6ba80ee2cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8349f2a49b9231f7c4e61632f7b22ed4
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 49, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_37a94d9e707cd865f217181b17391e87(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[1, 0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8, 49], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7220b5671cc1fbcac45e35bbf552ec91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37a94d9e707cd865f217181b17391e87
    def get_inputs(self):
        return [
            paddle.uniform([8, 49], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0877b5531e32f33a383f53bd157ac37b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[1, 0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2401, 8], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e1ab2ca8418af3715f653dacc5b8b334(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0877b5531e32f33a383f53bd157ac37b
    def get_inputs(self):
        return [
            paddle.uniform([2401, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d08fb43735e1af17a9cbe60dce5f95dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76dee8755d5dd20cf8c34a12ce9eb7c2
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6e1bfc6372783f01f98ed165cbe5693c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 192, 784], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_29d53ac7787bbd43eb9622fa7fb4f619(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e1bfc6372783f01f98ed165cbe5693c
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 784], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0bc02d9fc19e5175109a2be4a3ac0f5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c0688ceb8a47d57e27f99c90a19bb9b
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b93c975b04c598aa3c1e7ff8a4a2939c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 384, 196], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c32caff55573a02af858b93b65c88cf6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b93c975b04c598aa3c1e7ff8a4a2939c
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 196], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0c2deb8e07905eb4b1bb495ded91337e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 192, 784], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_912910cdc4ece2fb9cb24370b3ae5422(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c2deb8e07905eb4b1bb495ded91337e
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 784], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_795c512a7529c4b1b1c47810a8e3ce73(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6069, 4, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_47501913132e4084a387172bd79c1a4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_795c512a7529c4b1b1c47810a8e3ce73
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1c1ace306a5315e51ff65cececcdd8dd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 5, 1, 2, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 4, 8, 16, 8], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bb7d27fc1219bba3445c9a900672a4d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c1ace306a5315e51ff65cececcdd8dd
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 4, 8, 16, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7bd0d7befef9ab7751dfd29b0f1f2853(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb85ac39ae2f603bb1d17b3050053cd5
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 7, 4, 7, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2614844d310128078671981c398d53d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e09661471d5d54e80c75cb6b070a2ed
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 49, 3, 6, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e675acc5ce45e64abff54a2d7d27f17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dad816d24251c1ac6c20aecd26b8083a
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_18e9684df08c276a895a423b35d41928(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 5, 1, 2, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 52, 8, 202, 8], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_785dbfa983e07d0b269eb0ccbdea0f48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_18e9684df08c276a895a423b35d41928
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 52, 8, 202, 8], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0a44f2fce8f2b57e973c2ddf5c791ffc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 200, 304], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_085d92225802404ba88aeab295eaecd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a44f2fce8f2b57e973c2ddf5c791ffc
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 200, 304], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a8119ba02cc79389f478c9b76e99dc7e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 100, 152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9513b657909780f23abae6cdd63f9aa0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8119ba02cc79389f478c9b76e99dc7e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 100, 152], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7c56a7ddc3dcef04109fae33a4d070a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 50, 76], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d0e7f82b1be81b3c4914370ab9f11ae3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c56a7ddc3dcef04109fae33a4d070a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 50, 76], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ad7bd895e858e11d0d8ec59d02ea971b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 25, 38], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d1240742fd0339560d6adb906a74192e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad7bd895e858e11d0d8ec59d02ea971b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 25, 38], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_20aef1c1c84cbec37e6bbd57c401c94a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 13, 19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_74a279e9121adf195798a76cfca829e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20aef1c1c84cbec37e6bbd57c401c94a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 13, 19], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d149104814d252c747d4356628171ff6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 200, 304], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e0a079f628a2d6b48e37f0317a9ece27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d149104814d252c747d4356628171ff6
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 200, 304], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5ff3338aab435cc3f867c9ef65aa085d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 100, 152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e818bb2c2539f448feddc4b9be4db20b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ff3338aab435cc3f867c9ef65aa085d
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 100, 152], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c5bd9bc1d2804a7f9e93b749a3532a94(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 50, 76], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f18b76f9d92341adff4ebc5e8f63a95b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5bd9bc1d2804a7f9e93b749a3532a94
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 50, 76], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a4b3fd43645a9cce8b6ed1de35028392(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 25, 38], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ab1678617d847b6e846621a8c7d1a003(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4b3fd43645a9cce8b6ed1de35028392
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 25, 38], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4f2f15389b0f1b5b9ccf49161f4b5175(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 13, 19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3aad156a2e77a44f20d30d6c89a2bba9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f2f15389b0f1b5b9ccf49161f4b5175
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 13, 19], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7ea56368bb6e235ed3c1867df98613fa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 2116], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e4d0337fc546103c9a574fcbcf9f7e1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ea56368bb6e235ed3c1867df98613fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2116], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fb542bc2349354f902744e4524524f51(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 2116], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8b9c3422ed3b31bcf949d437375a08f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb542bc2349354f902744e4524524f51
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 2116], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_118a9be2ae1b1e1e95d791bb9dbc0206(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29d61a42275d6a4289b8be3acc810c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_23d6fefdecb5d88e4dd8557218f63671(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06f827971a8f84f612d65cc80006e776
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_017470034610b94343ae0c0b5333fa97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d0617dcf33c4fe3b1979074e17e5dbbd
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6055843f922f66b440362032a2059c55(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1025, 3, 6, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_145f302ad12e38ade89cee410daf396f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6055843f922f66b440362032a2059c55
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 3, 6, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_217d63f7958073d83b51c4fb61ee88af(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 1025, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7a26310bc101a98d06d17936eeed9127(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_217d63f7958073d83b51c4fb61ee88af
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1025, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_175a4a01163d8bbe50ef540cf4707016(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 1025, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d7f8ff116c70e853dae2e5caaf465da9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_175a4a01163d8bbe50ef540cf4707016
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1025, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_da8e2f1f93670ddd03038ddec1b687ff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 4096], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1edfa2e4d604a801cd1faaf59673e7ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da8e2f1f93670ddd03038ddec1b687ff
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 4096], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_647e2a486d5af368db11f7223957dfe4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4096, 4096], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cbaa4cc33c2db68b4c836dcb8664ddf1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_647e2a486d5af368db11f7223957dfe4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 4096], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e62cede13c47b0fd075f7db28cfa007a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 196, 8, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c9d110ddb37f906e41151cbbd40fba34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62cede13c47b0fd075f7db28cfa007a
    def get_inputs(self):
        return [
            paddle.uniform([22, 196, 8, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_acceec7afb49cdaace12f80dc01a26fc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 196, 8, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ccbf047228a311eddc037b7e2671eb10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_acceec7afb49cdaace12f80dc01a26fc
    def get_inputs(self):
        return [
            paddle.uniform([22, 196, 8, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_385fd542618f80dcbfc9567baaca9e40(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 24, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f1d968e93d263fbe1f078d5c84e4a7f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_385fd542618f80dcbfc9567baaca9e40
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 24, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_867c60590970bc95bb0e4fb82b534813(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 2, 24, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c51f6da38675f4f8d5f6de7d12507107(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_867c60590970bc95bb0e4fb82b534813
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 2, 24, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_79155854e8f8ff91e499fb119522bbdf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 24, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ca52d91c47352fa57788118b80b535fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79155854e8f8ff91e499fb119522bbdf
    def get_inputs(self):
        return [
            paddle.uniform([11, 24, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e296a6009d56824925aec9ebea84bfcc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 16, 64, 150], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0485ee993f6eca5dceefa59b0c045639(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e296a6009d56824925aec9ebea84bfcc
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 64, 150], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_51f6e4e493661b6052e041f3ff5dbbbc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 96, 3136], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_176e9358909137c69194ec61f8bb4733(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51f6e4e493661b6052e041f3ff5dbbbc
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 3136], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0127ddc84ca5f21b4b5cb37e04299df9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 136, 160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_47861fd768afb690260ba2570e8c0467(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0127ddc84ca5f21b4b5cb37e04299df9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 136, 160], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f7ef05586058d49cce8b09628e3017f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 68, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_23efd8a1774481e81a8f3af1d9e55e27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7ef05586058d49cce8b09628e3017f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 68, 80], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a4d7e8f3b02c4908d9f8552547fc80b3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 34, 40], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_789570b1ea0586b1abfccf225cf1dac1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4d7e8f3b02c4908d9f8552547fc80b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 34, 40], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0c067404a129af1d25fef40ec7a90b1d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 17, 20], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f7157857d0238a6e026d00da214b637c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c067404a129af1d25fef40ec7a90b1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 17, 20], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b27b308c03992b320510bc029faafa42(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 9, 10], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_01b2f167d4ed2c284b4e6ec6b7826c6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b27b308c03992b320510bc029faafa42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 9, 10], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_80d8ab86619c1daa187f74da05c003ed(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 136, 160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b711612330c60d911b5058219d641e75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80d8ab86619c1daa187f74da05c003ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 136, 160], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a80e54a615d54581e1efda91c569b40e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 68, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2fd0f6ac3a43a1afb261d36de7695f7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a80e54a615d54581e1efda91c569b40e
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 68, 80], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ce1cce585511baa0bafcc950a47abdd9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 34, 40], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_18ade3a04dcafd8ad00ad14828052f61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce1cce585511baa0bafcc950a47abdd9
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 34, 40], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ba90740b6b704d20ba50a6b6618e91e1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 17, 20], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4ca9aaf914e8e050afc45bc6e75ff53a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba90740b6b704d20ba50a6b6618e91e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 17, 20], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_af2b84fd51183f8c28f7b18aece0364f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 9, 10], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3f7f5f9b32cef233f8e0d77ef2d86c2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af2b84fd51183f8c28f7b18aece0364f
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 9, 10], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3059fa690e46dd0c435a708cacccebfc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 4, 2, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 16, 512, 8, 8], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eaa7b1010749c20df631ab7fbb3ba823(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3059fa690e46dd0c435a708cacccebfc
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 16, 512, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1029a2284d87cfbe7248d1a28bb120ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 320, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_807381469fc52889831bddbaf98464a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1029a2284d87cfbe7248d1a28bb120ef
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cf3a7adb1dcb3f4463b302f62c61489d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 256, 160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ab0db88546c3349473441eaa380268d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf3a7adb1dcb3f4463b302f62c61489d
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_118a9be2ae1b1e1e95d791bb9dbc0206(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29d61a42275d6a4289b8be3acc810c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_23d6fefdecb5d88e4dd8557218f63671(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06f827971a8f84f612d65cc80006e776
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_017470034610b94343ae0c0b5333fa97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d0617dcf33c4fe3b1979074e17e5dbbd
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3a67230c3193347e0b6e1a807209b2aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 4, 2, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 13, 13, 512, 8, 8], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1f25083f70eb449867701c17820b065a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a67230c3193347e0b6e1a807209b2aa
    def get_inputs(self):
        return [
            paddle.uniform([1, 13, 13, 512, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ef692944fddf2b13c0706903f9030fe6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_861b55d23659bfb7e30bb07acda5fed3
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_407e86043743db80b886354782c8f9d0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 96, 3136], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4d8c425c849a5c4be4b138deeed99498(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_407e86043743db80b886354782c8f9d0
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 3136], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c9d2965e597467c28f2027a1a53f6276(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2048, 5, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9d9e67a18353911da7d118731c739689(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9d2965e597467c28f2027a1a53f6276
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 5, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_afd1763d877d653eb86068bdc685f7f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2048, 160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_794902946c896e4e01b85e4ac673361e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_afd1763d877d653eb86068bdc685f7f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 160], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_08ec2da630de3d215fa8244900f191a7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 160, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_81f9d7250899869a5c917f7c757bee90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08ec2da630de3d215fa8244900f191a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_69b39c89f22d7a91facb307198102f21(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 2, 5, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9f721be6cd1994c832a008ae7d71ee87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69b39c89f22d7a91facb307198102f21
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 2, 5, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f8e055a33b2cbc7fa131e59b957331f1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 512, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7a4bafb61bed676b2e010d4a4bdc0d1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8e055a33b2cbc7fa131e59b957331f1
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 512, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c5571f1e0c96bba395b00c3bf0cc49c4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 2048, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_00f03636dd619bf1fb3dc940de1307a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5571f1e0c96bba395b00c3bf0cc49c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 2048, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7903acec907903bca894e8f134486b89(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 8, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f47be9cd79f3e0c7eb81ba879de804de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7903acec907903bca894e8f134486b89
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 8, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2c71de12601557fccc010dfc3e979eee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 2, 8, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4a9e8a350c8c5a260d3e2122bdda6934(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c71de12601557fccc010dfc3e979eee
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 2, 8, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9f6e9a1a605b343f4fb73ceb3d2681d5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 1024, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4bc39643a5e627e0b26e953261c54d07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f6e9a1a605b343f4fb73ceb3d2681d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_02bc6a38dccb8646403b48d70f31e5be(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 1024, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_87aff59912e84031b4169d6daa15f8a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02bc6a38dccb8646403b48d70f31e5be
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_540e613b338922b91701499f99b9e168(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 6400], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c0365d487f4d38134a92c09f04cb12a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_540e613b338922b91701499f99b9e168
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 6400], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e8cf598e584f9548ca50f937fe44a818(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 6400], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3b187011aacdb83b48bf8a1a12548a66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8cf598e584f9548ca50f937fe44a818
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 6400], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f9989e3907c452051a2730f12d482d69(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 3600], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c891d1f702da98824c874d1d23914a5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9989e3907c452051a2730f12d482d69
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 3600], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3f80182cee7e4cff251e432c81bddca2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 3600], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_00473f4a68ebc571ac2904bf703e560c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f80182cee7e4cff251e432c81bddca2
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 3600], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fcb868fecd7ae97159831c1c4c3647eb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16, 32, 64, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e4bd4759d90463c06e8363b6c9a66a09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fcb868fecd7ae97159831c1c4c3647eb
    def get_inputs(self):
        return [
            paddle.uniform([16, 32, 64, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_121dec1a955a346d4988ede95bcb82ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16, 64, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6bc1f17b69cf7402e20716f3350cda43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_121dec1a955a346d4988ede95bcb82ae
    def get_inputs(self):
        return [
            paddle.uniform([16, 64, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_691aa767a7b2c596e448604ce5f0e91d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 200, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4417972e149d9d6f674bc2b89f12dba3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_691aa767a7b2c596e448604ce5f0e91d
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_42f5361d1920d72c3ac9124b0c96d727(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 128, 100], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_686d997a718672d032f844c04decd77b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42f5361d1920d72c3ac9124b0c96d727
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 100], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_414776fa0a6a5de36682f642a2df1367(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef6eca8324cb89c425ead0be71e45bdd
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 6, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_98a4dde342bc73ca96d006e07e9af545(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb9f70282899345810179c07997ffef4
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5708d33f552d9b77709714e997a74d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4dc2635fc07050ef006a0bdd589c884e
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 49], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c6d1b02dfe21bbca641d45bc75fde578(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2945ca61ad8e249c717d30897e1fd40f
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 2, 6, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_27adff2a34754b179e250911ce9593af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_765dea6d119b72d25e404af98eb9718a
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_eea972cf53ab0bb8387ff456c760dc7a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 20, 3136], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_64e0664e57c4ac8e63e02d817126c6aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eea972cf53ab0bb8387ff456c760dc7a
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 3136], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_542219d27cf4f2511c63e867c4eb6efb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93fff2ee76cb12b87038ad024d7858eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 3136], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_40e713acc4060b0b8cbf4973a1cc0b39(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 9216], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8f1daa1d24f492d28fc49ce131ac93d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40e713acc4060b0b8cbf4973a1cc0b39
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 9216], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e9c57a07353d6475f7774238eb7219a9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 9216], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_92b1cb6f77391c53e820f629798a1744(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9c57a07353d6475f7774238eb7219a9
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 9216], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3044f5675935f19c9e65eaa393ecf842(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 9216], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0a48053fd8ed1c67364c04a0a630b874(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3044f5675935f19c9e65eaa393ecf842
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 9216], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f961ddc0424644abe0ad67633a6d981c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_513a7aa4a27595b824e8e952e1bb0c0b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 2704], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8c58840bd1568a7a9a6f5eb348647529(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 76, 2704], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9e2166f4b611cb35a2906d9bd0efaa97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c58840bd1568a7a9a6f5eb348647529
    def get_inputs(self):
        return [
            paddle.uniform([1, 76, 2704], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b894984190ad9dcad0250e238146175a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 232, 16, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7622373c67e151ada8945d3776d9dab4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b894984190ad9dcad0250e238146175a
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 232, 16, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_93cbe1276b633946cd197eaccf209986(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1247e5974bcf831b85bff78035225f53
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 7, 1, 7, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1dd6deb8b50611947da83583a90339b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92dedaec40aad3f7c347bb59c85ff802
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 49, 3, 24, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dd241240974bc91778db703596d2bbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5956615907214bfca4df75c5c979ad5
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_610831900cc4cf581a3b2e9fbeca972b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 197, 3, 3, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5f0500f8ed09ed3c540daa48dce49d4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_610831900cc4cf581a3b2e9fbeca972b
    def get_inputs(self):
        return [
            paddle.uniform([54, 197, 3, 3, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cb146fd3cb42137c92739798fb7bde42(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 3, 197, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d917a773df16ef8092075572250536df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb146fd3cb42137c92739798fb7bde42
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 197, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b70ec08e41922dc9bca2a540adbd9f52(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 3, 197, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8aa89a104cd0e72746a0b9658b78803e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b70ec08e41922dc9bca2a540adbd9f52
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 197, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_92df7e6af3f814c7ae5356b8a996b553(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 16, 128, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d9e50b17b2eb71158590018cf72a2a8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92df7e6af3f814c7ae5356b8a996b553
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 128, 256], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a4807b5e9d46cce74dc40e3066bd70d2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32, 32768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a93463ce43ca242dfb942c42c14bd5f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4807b5e9d46cce74dc40e3066bd70d2
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 32768], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_241129860d058742f15ce0d927f93967(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 65536, 1, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_62d5c838397c2ad1e060bd91a9ae51f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_241129860d058742f15ce0d927f93967
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 1, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_24345d51310411b8465cfb91719489fd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 65536, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_73411256edc26d48d9b04f55fda6e4dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24345d51310411b8465cfb91719489fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8a3fa4ba8994e7ff420de0e1e4164b40(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ddd88edee13b84fb7f6a0eb0daf805ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a3fa4ba8994e7ff420de0e1e4164b40
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_23d6d0fa59156c0558d648c028e9fd8a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 2, 1, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4a0207585a7fd4475ada4e147d861c60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23d6d0fa59156c0558d648c028e9fd8a
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 2, 1, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3e72216b8e093175a8d8f7f4f877cd11(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 1024, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d481ebada8be973967f9a528cedd00dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e72216b8e093175a8d8f7f4f877cd11
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 1024, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d6f6ea24e5909df5ed6265aec23795ce(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 65536, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ef3cb7998f0dc51965e0150282befb84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6f6ea24e5909df5ed6265aec23795ce
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 65536, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cdbda394e96ed1a5cac707ab2fe4e4c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1021a233ed67b23276c8f0415e98e21d
    def get_inputs(self):
        return [
            paddle.uniform([300, 256, 49], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ae050a5234f3f9a130e14111cddeaa90(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 640, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ceb98e65d1f554593b4227ae15c96db6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae050a5234f3f9a130e14111cddeaa90
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_59e04fe6a558f2e5f4a8621c12f2cd54(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 128, 320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_79ddce3b7a2679b4e1886fc3a5a7eb73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59e04fe6a558f2e5f4a8621c12f2cd54
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 320], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_118a9be2ae1b1e1e95d791bb9dbc0206(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29d61a42275d6a4289b8be3acc810c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c0365d487f4d38134a92c09f04cb12a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_540e613b338922b91701499f99b9e168
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 6400], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1df100311b9b31b9092e2e1bc53cc5b6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 6400], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7fc39553bb18c47b0ecf7bed2dc48c4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1df100311b9b31b9092e2e1bc53cc5b6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 6400], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c151f3cdc2c5d99745e9316d94075701(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 6400], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9b903c2c97f35d3a724430cd4c1ef4df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c151f3cdc2c5d99745e9316d94075701
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 6400], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0dfc4dd5695b435a5c4fc23fc1e17f30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f6c6361dc5654339d27ac533307ae9c
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 169], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e0fcff71aeb00a2fdd8f9b0e06170db0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_398e3e4874eca6a3306119ffa4329d14
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 169], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d9c3194fb9648928281c02df6c81aca6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 768, 49], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bbd3868290194733b57c5da991a2d006(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9c3194fb9648928281c02df6c81aca6
    def get_inputs(self):
        return [
            paddle.uniform([11, 768, 49], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_15a821c2e9a3ce80a5f70e762c6b6a27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_343f7d8725b39b02a9cffe0ced47c44c
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 676], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_341bbfa11050d0985660b1721a2a2775(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 676], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ac27f022bb6cf2fa8db8d21a32c60693(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_341bbfa11050d0985660b1721a2a2775
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 676], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3c949dac7a58daf808d01015f8193749(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 529], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0bb42afd2e169e7808fdc0481686b93e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c949dac7a58daf808d01015f8193749
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 529], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d205d0e5c4f3c6e2149dfbcce0202fe2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 529], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_663c180b5e262633c7b4ea6ae41958ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d205d0e5c4f3c6e2149dfbcce0202fe2
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 529], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e30b7c094389d9f78baba55030edf63f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb65fe6d53f3d62ff76d7cbac56b040
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 7, 4, 7, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0f336e1968bad2e272ce1799ddcab1dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_329d5670637f8070db71c773aa95c706
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 49, 3, 6, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e909c523570bc891756685db3c798f1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3412bda6865389dff7f108ee1309863f
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7521ee2398b8703704cdeb8d405b74a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e619824f41249824280889dd4e6db702
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_55657fb5204d24fac794f6677ae76fae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7fd087e9b8d337f4e1b6470e74a6e682
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10df1546204e938efab6a68c766e4a9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a65e69570ac7936e6fe370c04384258
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 4096], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_07a4640ff01771bdebc32673723129a4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8, 16, 32, 160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5475b2facc3d34b547961211da6b381a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07a4640ff01771bdebc32673723129a4
    def get_inputs(self):
        return [
            paddle.uniform([8, 16, 32, 160], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9659fcd7107ebd2aabbf1af5b9206c71(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8, 256, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6f693f57fb6452f1bcf5b525cd4b882c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9659fcd7107ebd2aabbf1af5b9206c71
    def get_inputs(self):
        return [
            paddle.uniform([8, 256, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5c6128d07d00a8e1e438ecafc18d2efa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8, 8, 288, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_624f9f0df0cd3574b89a6f93497fb64d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c6128d07d00a8e1e438ecafc18d2efa
    def get_inputs(self):
        return [
            paddle.uniform([8, 8, 288, 24], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9a8d3b5b266194bbbcd4030ef469ecc3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 1, 2, 24, 12, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_159bef91f906d086bfb80a686e72f54d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a8d3b5b266194bbbcd4030ef469ecc3
    def get_inputs(self):
        return [
            paddle.uniform([4, 1, 2, 24, 12, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e731d2e3c529ff0307c78684bca1591(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9874f711042e2bfd107011a52d643e65
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 16384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa5fab6332e1433b0403dc03e2d1b60c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_736429e2d6de83812117a67bd8bf0eec
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 16384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_169bc8da91ad15cf7be39a93e325cde8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c9267bec5ee55f9afae9c6d978b6d09
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 16384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e731d2e3c529ff0307c78684bca1591(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9874f711042e2bfd107011a52d643e65
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 16384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa5fab6332e1433b0403dc03e2d1b60c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_736429e2d6de83812117a67bd8bf0eec
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 16384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_169bc8da91ad15cf7be39a93e325cde8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c9267bec5ee55f9afae9c6d978b6d09
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 16384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b2c5397ab697908d4bcd7cc452f9f279(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c1527eec68163910f412fc52da11d63
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1600], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d49d2a75214fe20168073e6e4d4008b8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 1600], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a49682383c28ffecd789da2b39398400(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d49d2a75214fe20168073e6e4d4008b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1600], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_448bd9ec0f6b78585c8656363a91f603(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 1600], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f27ac92a41f8029e6c2385ddd47f3d76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_448bd9ec0f6b78585c8656363a91f603
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 1600], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0c0adc21bd253225d1aa215ee7d0cb63(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 72, 14, 25], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4a73ffc806a8682acf1f25bb10456a85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c0adc21bd253225d1aa215ee7d0cb63
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 72, 14, 25], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_400673a6d7a75cf3cf46e6fe57fb5fe8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ded1a7c05af155087ddb75b3ab878e1d
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 3, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ef692944fddf2b13c0706903f9030fe6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_861b55d23659bfb7e30bb07acda5fed3
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_da4ce96ab6c131153d81411b696a1565(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1c5ea8fe5310ea4b3cb7a0cda27ec5d
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 49], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0164606818aab1c09d695d6cd0c8afd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db74162f4e6a34eb35457836172edf90
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 2, 3, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d67028909bf9abf7ce21096e1404246c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21259e70f761ede775e12e07f6f854
    def get_inputs(self):
        return [
            paddle.uniform([11, 3, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0f288fbdc80ad394b57bad99af86f431(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50e179c887d95e472bab06e86b01d414
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_43eede5085a2c790494760fca87050a1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 384, 196], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dc1fdf189ef3e2f628e1d5f54ef08c52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43eede5085a2c790494760fca87050a1
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 196], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_de1aa28b87ffd8d1cba2ade5ca1a0ad4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 4, 5, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 8, 8, 128, 4, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9842fa6e4ed5495ba3b14bfe802a7ea1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de1aa28b87ffd8d1cba2ade5ca1a0ad4
    def get_inputs(self):
        return [
            paddle.uniform([4, 8, 8, 128, 4, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_272725b753856e4a5137f597237b3d67(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 4096], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3681d3dbf384e78ae151945202b31aa0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_272725b753856e4a5137f597237b3d67
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 4096], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6ab9878002c9d8f8f9432f54b7fc4017(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 96, 40], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8ac317845c3f5697a4801258722f46b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ab9878002c9d8f8f9432f54b7fc4017
    def get_inputs(self):
        return [
            paddle.uniform([10, 96, 40], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ad02fa41dca084fca23a2efa148a5ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_595f21ae57133de93b9d170c3981407a
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 3, 4, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c11048827d52206d6b94ca26cce54cef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_768c972b03a389502e72873a3e86a2fc
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_601f6fe9d2c4f02398643696a25312ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccf813327c1df683ffd36b417520bc5d
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_96638f7ff96debaf0768f3a63848e10a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 361], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4333fa6b3358051583156db1d1d7f56f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96638f7ff96debaf0768f3a63848e10a
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 361], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6a1d170e58c557fc317d00b260768bbc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 361], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ccda0d54ad4ad27efd47667205868c1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a1d170e58c557fc317d00b260768bbc
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 361], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_118a9be2ae1b1e1e95d791bb9dbc0206(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29d61a42275d6a4289b8be3acc810c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_23d6fefdecb5d88e4dd8557218f63671(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06f827971a8f84f612d65cc80006e776
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_017470034610b94343ae0c0b5333fa97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d0617dcf33c4fe3b1979074e17e5dbbd
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_896397393d793e1d2f2923c398dbeb5e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32768, 1, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_362dbebb8e8486cb20711b4058d5ffe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_896397393d793e1d2f2923c398dbeb5e
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 1, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bf63678c9d687f747a1cf5350c65dde3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32768, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ada7fac199f3b376370ab30ff0d88d76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf63678c9d687f747a1cf5350c65dde3
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_978eb3f04d7b77bead5085990d1adbec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_52bf600e6a68a77a05f8f66917420ef6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_978eb3f04d7b77bead5085990d1adbec
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f99674a401e2d83af47f86f1d561cdb7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 2, 1, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ebc52c5e5f32914b191757c134738f0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99674a401e2d83af47f86f1d561cdb7
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 2, 1, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_89bdea545680588d30440fe170697669(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 512, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b7076340f4b92112a5e2d455a656cdde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89bdea545680588d30440fe170697669
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 512, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1979a4feb0c4b787bd6b220ce545ca80(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 32768, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_96234089e831d31886f8b768f372d88c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1979a4feb0c4b787bd6b220ce545ca80
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32768, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f1d968e93d263fbe1f078d5c84e4a7f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_385fd542618f80dcbfc9567baaca9e40
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 24, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c51f6da38675f4f8d5f6de7d12507107(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_867c60590970bc95bb0e4fb82b534813
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 2, 24, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ca52d91c47352fa57788118b80b535fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79155854e8f8ff91e499fb119522bbdf
    def get_inputs(self):
        return [
            paddle.uniform([11, 24, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0485ee993f6eca5dceefa59b0c045639(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e296a6009d56824925aec9ebea84bfcc
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 64, 150], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9cecf660b596c07ef2e1ad1015291353(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d1c913ba21ce1e30fdc8dfb9f83fc71e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9cecf660b596c07ef2e1ad1015291353
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_23d6fefdecb5d88e4dd8557218f63671(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06f827971a8f84f612d65cc80006e776
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_41ee67b03399b8077d7c6336443a4403(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_82535c6f4bc4b77a1fab9104606eb8f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41ee67b03399b8077d7c6336443a4403
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cb1fe9c9c86b255a4d69a38bbf6719b8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 200, 3, 2, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d076e063b75ae45c59ee340c42aaa7ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb1fe9c9c86b255a4d69a38bbf6719b8
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 3, 2, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_20ae0c1569c8c951b6f52ba05255c8da(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2, 200, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ee039f56b5525644e7e24894f1822e66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20ae0c1569c8c951b6f52ba05255c8da
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 200, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9a68846c5d3935bfd0061e1e16314a27(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2, 200, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cbd72d75605e357b9d7f5be0db1107b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a68846c5d3935bfd0061e1e16314a27
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 200, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fc060999287e06fe5e62ec51a3a39d4e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9261, 4, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_521780c69c05d9c6d7dcbbfd55e62ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc060999287e06fe5e62ec51a3a39d4e
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_049b618a25a7043d0af5276b85d4902d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 16, 16, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9483d5c7afb676476c13790ea04f63b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_049b618a25a7043d0af5276b85d4902d
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e3206f7185afb1dc5e0a64266fb51b51(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[1, 0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16, 49], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_87d36c4ff337c2095183b13f2ec0257a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3206f7185afb1dc5e0a64266fb51b51
    def get_inputs(self):
        return [
            paddle.uniform([16, 49], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7ba9da62a85a06a1c0f2e65f20810e92(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[1, 0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[784, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f1508607cddcb284048093f65780d375(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ba9da62a85a06a1c0f2e65f20810e92
    def get_inputs(self):
        return [
            paddle.uniform([784, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d6a304c8d2dc4ffe90823b3921654dd0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 16, 49, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_772a95dda3d30a6d8f24a319eed10c8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6a304c8d2dc4ffe90823b3921654dd0
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 49, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_55cc5bba00b9cb3fdc64e55f67d86fe8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a76b4b8cafc95d696c297cf3ee0c5ac1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55cc5bba00b9cb3fdc64e55f67d86fe8
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4e67ca82bb4d9ab8bdb728d4bfb068c7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 197, 2, 6, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3b142a7dd31f9abedd72f47c5d0e1be9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e67ca82bb4d9ab8bdb728d4bfb068c7
    def get_inputs(self):
        return [
            paddle.uniform([22, 197, 2, 6, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2a8afe6affb6234c3f0117b21a0a33b5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 197, 6, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4c3ac34a7f6525a0c7df780a871f3b1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a8afe6affb6234c3f0117b21a0a33b5
    def get_inputs(self):
        return [
            paddle.uniform([22, 197, 6, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f35a4b9eaa6098453feb4f89ed9723f7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 6, 197, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2d22ceba14534775277160f2d40a03af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f35a4b9eaa6098453feb4f89ed9723f7
    def get_inputs(self):
        return [
            paddle.uniform([22, 6, 197, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_70e8e4ab314e9a804533376c74611d5b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 100, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2f3a2141e1aefd8de33311ebdf724ca4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70e8e4ab314e9a804533376c74611d5b
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cccc826efb628b41f87775e3944f9f13(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 256, 50], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_11aae7353f21629ff141ab1879a4df69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cccc826efb628b41f87775e3944f9f13
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 50], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0aff650352063f433d7f9b68fdca4414(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 21760], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_60d5ee5342313e2b557c455a04546f8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0aff650352063f433d7f9b68fdca4414
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 21760], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_70aabea4fe49d7d2c3efb61aa3a1b6d3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 21760, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_09b017f2ff9cfe9715abc1f35757f4e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70aabea4fe49d7d2c3efb61aa3a1b6d3
    def get_inputs(self):
        return [
            paddle.uniform([1, 21760, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_60d5ee5342313e2b557c455a04546f8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0aff650352063f433d7f9b68fdca4414
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 21760], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a112a2636ecca3ca355dff12945f8dd6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[240, 4, 96, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ff4f7baf43f43f756d5cc54125ceaefb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a112a2636ecca3ca355dff12945f8dd6
    def get_inputs(self):
        return [
            paddle.uniform([240, 4, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_66c6c6b526ade27b1a77f4afbe02a176(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 1, 24, 48, 2, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_243a5c34bb02ec6c8fca3b69481e76f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66c6c6b526ade27b1a77f4afbe02a176
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 24, 48, 2, 96], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_da6d9138c3efe0b5c36c28c92c2de891(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 32, 144, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_619a78914dac4792e42b66ada1021863(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da6d9138c3efe0b5c36c28c92c2de891
    def get_inputs(self):
        return [
            paddle.uniform([4, 32, 144, 24], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d70d9245391e82dec3bf9cdaa633bb2d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 1, 1, 12, 12, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_570e61aa55b1f5c164f25f93b481bbe7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d70d9245391e82dec3bf9cdaa633bb2d
    def get_inputs(self):
        return [
            paddle.uniform([4, 1, 1, 12, 12, 768], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7f354d453a96c3212e0e63fbfbec637d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 136, 208], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_21de331709aec87cb1511ab8a29e83b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f354d453a96c3212e0e63fbfbec637d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 136, 208], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_65e8a94562e24f9b760483f6677fdd22(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 68, 104], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e1dac03fb9d2608bea9089f29c75db61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65e8a94562e24f9b760483f6677fdd22
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 68, 104], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e1e37094676421b43042f70c28c7d013(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 34, 52], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_880f8679e8f38ebc8d58e5eb847706f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1e37094676421b43042f70c28c7d013
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 34, 52], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c8ff8def20d256b50ebd7230895f68c7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 17, 26], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eeb321fa8d2002b38a05b6aba32c9790(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8ff8def20d256b50ebd7230895f68c7
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 17, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_011a7d3243ea07f736899fe8eac5cc73(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 9, 13], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_10ef61707cab998c456e7d9a7d5d5ddd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_011a7d3243ea07f736899fe8eac5cc73
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 9, 13], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8e7c249acc73fb35ef843c5eb6e18cac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 136, 208], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dee6d407ce0d93f1ccbcdfd5ad215892(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e7c249acc73fb35ef843c5eb6e18cac
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 136, 208], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_838dd935d889cf5b1aacf0fdf3895208(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 68, 104], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eeaec5a0cf7967f4147aa5da0567f939(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_838dd935d889cf5b1aacf0fdf3895208
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 68, 104], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c50dcca807362ab8d230bfa169092ab2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 34, 52], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1d573e88519fa7136e14f3d2afac7ca8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c50dcca807362ab8d230bfa169092ab2
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 34, 52], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_658c940c86e5edf0b7a1c42d3ea533e3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 17, 26], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_80d8bd3a628b082635d97ae7e71607ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_658c940c86e5edf0b7a1c42d3ea533e3
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 17, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bf41a6cf07803d7a86adfcb894f91b32(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 9, 13], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f10a88310aa104165c77f6dcc37f12ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf41a6cf07803d7a86adfcb894f91b32
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 9, 13], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_15a821c2e9a3ce80a5f70e762c6b6a27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_343f7d8725b39b02a9cffe0ced47c44c
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 676], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ac27f022bb6cf2fa8db8d21a32c60693(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_341bbfa11050d0985660b1721a2a2775
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 676], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0b81e24bb92111aa18dc7e4b62ed85c2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 2, 7, 2, 7, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f7129fbee81913ed9b83281cdfdc7215(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b81e24bb92111aa18dc7e4b62ed85c2
    def get_inputs(self):
        return [
            paddle.uniform([43, 2, 7, 2, 7, 384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_58d46d6af14fde8039b960580dddc065(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[3, 0, 1, 4, 2, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 4, 49, 3, 12, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0308039b85b4325115901cfa6c6461af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58d46d6af14fde8039b960580dddc065
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 49, 3, 12, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_35b94e436c33d86a3020111bc76e2ddd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 2, 4, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 4, 12, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_64f7415bf94b6c3fbff36ddcd72d805b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35b94e436c33d86a3020111bc76e2ddd
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bfb86674f08e2524dd57ca19edcc36a5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 4624], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_96ce4c0777c0749d36605d7ffb3b65ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfb86674f08e2524dd57ca19edcc36a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 4624], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_098eaf75e3bce05255ce6aadc66086b9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 4624], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f0ea82280cb7051bf21e05d6c419b724(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_098eaf75e3bce05255ce6aadc66086b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4624], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_79ce896732e27d96b289025f8096e116(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 4624], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_338acfb5835b01abaae08aa5ec4d0bcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79ce896732e27d96b289025f8096e116
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4624], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d85cfe50e547977d49050d9b08059f89(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8192, 2, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0abee9074e826f69a4288955029c99a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d85cfe50e547977d49050d9b08059f89
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 2, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_90e74a1e3bc05c253c6d6e4e5c864a84(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8192, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_802345e5fa42270d1fabb25e006dc8f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90e74a1e3bc05c253c6d6e4e5c864a84
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ae652015f939e052d55893056ce9a534(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a3c95aea9701d471289f37d81131c16e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae652015f939e052d55893056ce9a534
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_20157e0824511b991b9e7549078c6ba5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 2, 2, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b68d7b10931399b89b1cf26695cc8cee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20157e0824511b991b9e7549078c6ba5
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 2, 2, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_22294cb54e2821625b050ae36c35e47d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 512, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3a787cf53d902456f4a1537bacc024d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22294cb54e2821625b050ae36c35e47d
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 512, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cd2b0959929de0be1acf5d2a74d1d5be(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 8192, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d7b54729ed4262f6ade4c7a9fd5048e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd2b0959929de0be1acf5d2a74d1d5be
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8192, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a2b623cc35b19d5fe9deb4193183accb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2048, 5, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_490759e5ab3e766ef965bab1c6bf08af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2b623cc35b19d5fe9deb4193183accb
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 5, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0465a19f49de58d270a5580e7a47e3f7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2048, 320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bff9c641ab8ad31b1df688207a8fb722(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0465a19f49de58d270a5580e7a47e3f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 320], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8b4eea8dc94052ea30a67cf56c0ba415(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 320, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1a91257cca89e8fd1572da3e56cccbb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b4eea8dc94052ea30a67cf56c0ba415
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a1428b8506f61ffc3f32984877524fca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 2, 5, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_579bce8f0b4e08f7b6b5d05afb618c15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1428b8506f61ffc3f32984877524fca
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 2, 5, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_baebe8b3fe9a9591180b5400d5cfbc75(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 512, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_80fa9e3dd8758e8d3edca14a9618fbe5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_baebe8b3fe9a9591180b5400d5cfbc75
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 512, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2595617f032e63ecd1be0445db7e4c59(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 2048, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ec7039039d9e2df487cc9073b87f0db5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2595617f032e63ecd1be0445db7e4c59
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 2048, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_566c2904724b532696c4caefaf816cd5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 20, 1600], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ccad4a6056d126e573037411d819ac76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_566c2904724b532696c4caefaf816cd5
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 1600], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_48b02826f3fb3b3e95bc50c86413d041(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc4e16bdca22ccda1ee9297bb7762fd5
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 1600], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7e6b0473d74e0699d5e2d24152598a65(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 5184], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e850ea63917ec52bf6e78d868f2be8a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e6b0473d74e0699d5e2d24152598a65
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 5184], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_48550eb47e241be7e5b76587382f5255(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 5184], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0c6b971b154cf327d7019f41a4f39677(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48550eb47e241be7e5b76587382f5255
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 5184], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3d43840fb8e9931b68edbf8df54b777e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 5184], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e8a30df920a7ed1c59eb44b1d566b586(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d43840fb8e9931b68edbf8df54b777e
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 5184], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8ce519591fb11496bbf51928e75c4ccb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2100, 4, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0f5ab8ab8c44273bb86c073ddc22ae84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ce519591fb11496bbf51928e75c4ccb
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1430289c7531fc62ba5efa5888b7e0d7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 8192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f10b5034090e45af2b606c848ea0b8cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1430289c7531fc62ba5efa5888b7e0d7
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 8192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4adb6a64a0ae43ada89686a8a2304d9f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8192, 8192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2ffd9197fce274f68f20e7b2b163e736(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4adb6a64a0ae43ada89686a8a2304d9f
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 8192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2c5019a283b003b024cc7047402fc7dd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 16384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bfc7c8f6ffa1a1c481c32b7d187a5574(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c5019a283b003b024cc7047402fc7dd
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 16384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_636e2dde81af971a75ff84073e796eb9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 12, 12], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_92063f25dbf720b830e4a88e6719e02e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_636e2dde81af971a75ff84073e796eb9
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f70a1bb3d43b1e8da4056892e6a221e1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 20, 100], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2cdc6a54802bec83a38437887803b269(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f70a1bb3d43b1e8da4056892e6a221e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 100], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_57dc0da21db17ac38030326ddd715dfc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 100], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_271b450d1e49ce16a702ca846b1d7ca3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57dc0da21db17ac38030326ddd715dfc
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 100], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b1d629783b007af804fbe95a73f06c4c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 1, 7, 1, 7, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0db17834bac87158839273404a8e0cae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1d629783b007af804fbe95a73f06c4c
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 7, 1, 7, 768], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_77a7e7996acbc6fcd97f4e0f2af9d257(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[3, 0, 1, 4, 2, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 1, 49, 3, 24, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_96dbeed38a9054c5aeec86722bdeab0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77a7e7996acbc6fcd97f4e0f2af9d257
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 49, 3, 24, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5ab7b62f825fe7beca19a9aa43990acc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 2, 4, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 1, 24, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9fa620d5aad76de940039a84f88b46e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ab7b62f825fe7beca19a9aa43990acc
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a76b4b8cafc95d696c297cf3ee0c5ac1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55cc5bba00b9cb3fdc64e55f67d86fe8
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bbd3868290194733b57c5da991a2d006(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9c3194fb9648928281c02df6c81aca6
    def get_inputs(self):
        return [
            paddle.uniform([11, 768, 49], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0e35fbec8aa18f5da5f08a3430d2c6fd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 96, 9216], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d2a9814dccd832a4015bfcf912674c6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e35fbec8aa18f5da5f08a3430d2c6fd
    def get_inputs(self):
        return [
            paddle.uniform([4, 96, 9216], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_976f2ab0cbc1cb1de6413db5cc26e099(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 20, 400], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_186b2c96c668c0f7441a441b759f4587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_976f2ab0cbc1cb1de6413db5cc26e099
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 400], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75abc81b00c0d0b962b6387f7e95f4b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c51269703391cb3c9db022ed4a74d00
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 400], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bfd3f30f3ea72d2295750820bd79414d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 2, 7, 2, 7, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_54ac8a8bb2d1e6723528fe7f99d7f219(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfd3f30f3ea72d2295750820bd79414d
    def get_inputs(self):
        return [
            paddle.uniform([11, 2, 7, 2, 7, 384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4823e8f104c33c1b0f5551ed26ca94bb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[3, 0, 1, 4, 2, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 4, 49, 3, 12, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_708b6b55288d8470172351af8127bd36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4823e8f104c33c1b0f5551ed26ca94bb
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 49, 3, 12, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0487eb4fba7cdca9eae703b7ccacb830(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 2, 4, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 4, 12, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2be58405f50271cd9e14a1622be17ff8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0487eb4fba7cdca9eae703b7ccacb830
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0db17834bac87158839273404a8e0cae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1d629783b007af804fbe95a73f06c4c
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 7, 1, 7, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_96dbeed38a9054c5aeec86722bdeab0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77a7e7996acbc6fcd97f4e0f2af9d257
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 49, 3, 24, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9fa620d5aad76de940039a84f88b46e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ab7b62f825fe7beca19a9aa43990acc
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0d414373489457c5158a516adb691674(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1025, 3, 12, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_50b1ef58e648467ef6c4e80d94e7389b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d414373489457c5158a516adb691674
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 3, 12, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b439e2092c7de8d7c3138eff0f569baf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 1025, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a8acbcc3f1fe42069e41a02b51096325(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b439e2092c7de8d7c3138eff0f569baf
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1025, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cfd15314c427d1dff979b48b33d8a256(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 1025, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8b2a34cf637c288275e8f7d7cf387046(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cfd15314c427d1dff979b48b33d8a256
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1025, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e1bdc423af734e1ea8207f8a0f7eb1aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 768, 49], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_31835b63757a48a21dae495d0de799ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1bdc423af734e1ea8207f8a0f7eb1aa
    def get_inputs(self):
        return [
            paddle.uniform([43, 768, 49], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e7daff024cfaac9ff6127c6681996bd5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[44, 8, 288, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bcb7674a86cc63fde0124c0abe3c44a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7daff024cfaac9ff6127c6681996bd5
    def get_inputs(self):
        return [
            paddle.uniform([44, 8, 288, 24], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c9cfdff6d2027ad44ce5f1bb978aa35f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 1, 2, 24, 12, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_84317cc83e675d5a97ae8b5124ca71e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9cfdff6d2027ad44ce5f1bb978aa35f
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 2, 24, 12, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_54ac8a8bb2d1e6723528fe7f99d7f219(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfd3f30f3ea72d2295750820bd79414d
    def get_inputs(self):
        return [
            paddle.uniform([11, 2, 7, 2, 7, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_708b6b55288d8470172351af8127bd36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4823e8f104c33c1b0f5551ed26ca94bb
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 49, 3, 12, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2be58405f50271cd9e14a1622be17ff8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0487eb4fba7cdca9eae703b7ccacb830
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_470489840a6126a06e53c706c1a1146f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_581351d2c9f222d4e806d2f588dfd829
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 576], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6bbe1fab07a2d8b77bf84006036cd112(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6f4128b5aa91d80541ab4f5fcb6f9e9
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 576], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_80ec024025df99aef42716943aed939c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eb7263b6f7a302142bae47f5a65e796b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80ec024025df99aef42716943aed939c
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_612087630e0dcc2c72af6775c3d9003f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e8d9b63e2c91b585c5fed809877a683
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 20, 128, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_719b418e1a34bf6366cdcd24fbeb1154(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_892f8c36460cc78452c2435bef822a26
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 40, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_23b4cd10e1ef8ee982a07ee0d5b70e73(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 80, 32, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fd6c70c47f3c516e2f0f5b346fa410b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23b4cd10e1ef8ee982a07ee0d5b70e73
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 80, 32, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7b4b5f2b67b5b35904c6990124782d74(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 160, 16, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_52d164d7bc818303a4cdd279d87dea0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b4b5f2b67b5b35904c6990124782d74
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 160, 16, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4cb8c42b1f5e23c90aaafb60b7fc266d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48a4ff4883b2b1c63f9403eb1f027eba
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 176, 264], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_734bdd27474d456855bb2c9b0307d3f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db10f8c5ecad126da2b59f0af899623c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 88, 132], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7ea532a292bbe3c4d743a606ad2cd582(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_915e098e1d32d8d935f3b03b0f1535c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 66], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_65084e7185d3a7656e33f973c2ccb0f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ee36dff715dbbb15f0943d5f1e0b8ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 33], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6c209b1a077e3b3c70384450474a502a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 11, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_938a04392967fa09a30208e3cd74e574(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c209b1a077e3b3c70384450474a502a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7b41ada1e7f77928a608a5ebd3867ff0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b11306caa279b647442d9aa76c4636b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 176, 264], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_39a8212ccb03e6c24d717ec33f145062(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd86645fd2e06be3a4690944602ce701
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 88, 132], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d54ad24cff2b5235b95bb7bfdd826079(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89a655b83039e313be97e4657a80d300
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 44, 66], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_74363fd005d51d3d0a0abdfc6c67c498(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75508c87eafbc3169991cd511bf194a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 22, 33], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_268ec478354039d6eb264fc6caca2726(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 11, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4dec01fd8d724aea0fc3eeb7250dcf49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_268ec478354039d6eb264fc6caca2726
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 11, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5da5002f6ae0fa2d838355debc603a69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2420ed4590915e918010772c890c36c
    def get_inputs(self):
        return [
            paddle.uniform([100, 256, 49], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_514ad0e9d1f84538b58ebc86e236420f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 8192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f9368cab10eab4dba185e9590c399a7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_514ad0e9d1f84538b58ebc86e236420f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 8192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c32caff55573a02af858b93b65c88cf6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b93c975b04c598aa3c1e7ff8a4a2939c
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 196], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c5c3865ed6f5d8d4274bfcb2cd2d2d5c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 8, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ad4b46ef22e7da5df8e81b210972547e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5c3865ed6f5d8d4274bfcb2cd2d2d5c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 8, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a7c24eb603b2bf25499d4320613bf8f2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 2, 8, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c39eee331e2d24e7ca408db6854ad6d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7c24eb603b2bf25499d4320613bf8f2
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 2, 8, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c7460546426e275723552fc9e013621e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 1024, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ee0a35aeaeac4e08c198c57f812bce79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7460546426e275723552fc9e013621e
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6f2bba2a672ed20c03ace2044e9549a0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 1024, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a6c5aa42812aaaad0f276b91635ef28b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f2bba2a672ed20c03ace2044e9549a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9a51c504386f459556682c37e6409aef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 11109, 4, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_23e54738e8b43b580d53799f3e9d86b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a51c504386f459556682c37e6409aef
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b378a7e57743b4e61da2407d39e0c526(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb7f956137ad38cc22acf3dc633d470a
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 7, 8, 7, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e586202c5c1056723578cea25de3b353(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76cefa5e02fef6152f89c889e46792ee
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 49, 3, 3, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3791670b2db9fee4ac81f89e2ff20c77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68bdbe2c601197f5a6528c3b85cf0f2b
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f69353983594ccdac5498f230ce602b3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2048, 1280], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3b8050830e1cde1ff8acd0e647c66137(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f69353983594ccdac5498f230ce602b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 1280], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7b96c8610810db813558a6f2cc0fa0f6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1280, 2048], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6c832cf9ecb28bc23324a04ec3f5cdde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b96c8610810db813558a6f2cc0fa0f6
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 2048], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a424ceed48a127613f4f0661535f605(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef34883d20f522e17d7e43049b0030f4
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_31835b63757a48a21dae495d0de799ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1bdc423af734e1ea8207f8a0f7eb1aa
    def get_inputs(self):
        return [
            paddle.uniform([43, 768, 49], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f7129fbee81913ed9b83281cdfdc7215(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b81e24bb92111aa18dc7e4b62ed85c2
    def get_inputs(self):
        return [
            paddle.uniform([43, 2, 7, 2, 7, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0308039b85b4325115901cfa6c6461af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58d46d6af14fde8039b960580dddc065
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 49, 3, 12, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_64f7415bf94b6c3fbff36ddcd72d805b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35b94e436c33d86a3020111bc76e2ddd
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f961ddc0424644abe0ad67633a6d981c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_513a7aa4a27595b824e8e952e1bb0c0b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 2704], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_123e342871eb23e593f4d14ff86ffafa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e25ce0e126b680a08732401d6e34c41d
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 2704], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e731d2e3c529ff0307c78684bca1591(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9874f711042e2bfd107011a52d643e65
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 16384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d5c131b0034a70be883dd3b4ab0c3fbe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 11, 7, 7, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d0eabe0ae4667d592a1b1295460f22f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5c131b0034a70be883dd3b4ab0c3fbe
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 11, 7, 7, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ce6bc6ee59e259d9d2538933c20cfee3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c69235fd3d31c63159f07f5ad042bf71
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 24, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bab23deca8c0edfcf5f56d5b9e45f7d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fb31d094233c7b13fc7ef772520913e
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 2, 24, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5015c130c56d150aaecba54613056e5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c85fcaddcc667107ada757cc01b54fae
    def get_inputs(self):
        return [
            paddle.uniform([43, 24, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_aa64eec84a2c66d1fd8342a25636c318(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 192, 25], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bc1599856e5c82561624cd26865ddcc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa64eec84a2c66d1fd8342a25636c318
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 25], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_49c0ee6a1a57cc8338757edd3c26b2ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4dad0e76957d256859563cd800db89cc
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 400], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_db91778fdc4adc3b14ba7b400c495576(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 400], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c717bbeecc0153a2014eaad59068d3c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db91778fdc4adc3b14ba7b400c495576
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 400], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_db523dc799dcdb32406127300fde26e4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 400], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_19d8f382486e8dc1cac8098e975d73d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db523dc799dcdb32406127300fde26e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 400], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_34a7921ea0b6032b79b7c1a8af883d41(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 8464], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ccae6dc59f1861b7f56ae9caf9b26f57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34a7921ea0b6032b79b7c1a8af883d41
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8464], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ab1a86a3f2486242bdd22fff6765221c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 8464], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b403ee87087a2aa1080d75340e0be2e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab1a86a3f2486242bdd22fff6765221c
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 8464], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d24e171513e925cbc24a65ec586ad6c7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[144, 4, 96, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_85faa66ea47cf00251c7115e2d6ef463(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d24e171513e925cbc24a65ec586ad6c7
    def get_inputs(self):
        return [
            paddle.uniform([144, 4, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5a4da4b96353afe277f9325bb56f77d9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 1, 24, 48, 2, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6a2649476a02913bd85e6f026fa17b3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a4da4b96353afe277f9325bb56f77d9
    def get_inputs(self):
        return [
            paddle.uniform([6, 1, 24, 48, 2, 96], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2b11debf5b321ab92ec7dfc9ccb048e4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4096, 5, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_640b6a2ab1cd171e172a432374546a0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b11debf5b321ab92ec7dfc9ccb048e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 5, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_45ba505df996d16cf041e0fb52ccb03e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4096, 320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e7d4712109285ff692e19408de4a7f88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45ba505df996d16cf041e0fb52ccb03e
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 320], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ccfc078439db030d7e59582d2d2a258a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 320, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_24aa33629757f49f9031a0270e04e293(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccfc078439db030d7e59582d2d2a258a
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_08b66da46f82734be4b31de79a4fe81c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 2, 5, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c0e7359278c65f760c7b6d4b95bc1457(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08b66da46f82734be4b31de79a4fe81c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 2, 5, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cc628eefd68defb8c6e9f8772655281f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 1024, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_315963d63328f352e434834bbe765640(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc628eefd68defb8c6e9f8772655281f
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 1024, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_727d4678d3b855ac80d42a510e2e962d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 4096, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e1ad49967ceeaf28f04a40f4178d723e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_727d4678d3b855ac80d42a510e2e962d
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 4096, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9ec09d020c001820261942dcd6855aaf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 200, 272], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_25e2f7bc552ac0e857f9c22b4ed87ec3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ec09d020c001820261942dcd6855aaf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 200, 272], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_623ca32d6aac6840c4013ff7ca5391cf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 100, 136], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8b519dd4120e0e28e42f85269bbc1468(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_623ca32d6aac6840c4013ff7ca5391cf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 100, 136], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f1b57fe7b1885402e6716364ccf46eb1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 50, 68], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_38ab80d97d1612c4a34e7404e37047ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1b57fe7b1885402e6716364ccf46eb1
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 50, 68], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f626516335316c95f009dbe5c63580ed(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 25, 34], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_661f15e8b433d536f0a8467a3da168ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f626516335316c95f009dbe5c63580ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 25, 34], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0539fdda9fa45be3a2ca503eb32354f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 13, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ad0f4b6da64b8c3488d56e09c89bc2f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0539fdda9fa45be3a2ca503eb32354f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 13, 17], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_53d31089c17ed3772e935ab828695f80(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 200, 272], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7b6bea18870e3c9e9270425de84dbf99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53d31089c17ed3772e935ab828695f80
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 200, 272], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_114b14b31ebd06b19b373e9cdad89fbf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 100, 136], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f2eb21a316f2aed14b157eb75dec34f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_114b14b31ebd06b19b373e9cdad89fbf
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 100, 136], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cf9a81703bfb55a60dc625caddf0e244(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 50, 68], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_153fc13e948a0d6c78160859c744a073(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf9a81703bfb55a60dc625caddf0e244
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 50, 68], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9e72c184267002ddae13a60394f17ad1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 25, 34], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a325307db58ac083db7676c59cac09cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e72c184267002ddae13a60394f17ad1
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 25, 34], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3c69fc901782220a6e8d3f827023c5c0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 13, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_41eb12375df409ae9d8f09ba00042094(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c69fc901782220a6e8d3f827023c5c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 13, 17], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1f7e050a3307aa395695dc19afd93826(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4096, 5, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bae30a549c5ea539851606f11039aa33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f7e050a3307aa395695dc19afd93826
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 5, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7aa101bfd3643f9fe0d2072f09328eb7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4096, 160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_086c2163981ba91c160eebe2bafae8ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7aa101bfd3643f9fe0d2072f09328eb7
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 160], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8e2c45c42dd3dd2818e55e12e5e99954(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 160, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_22c3078243eebd41804b3a15ac65bdac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e2c45c42dd3dd2818e55e12e5e99954
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f27408dd083eb56f665338cc94361e6b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 2, 5, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6c402f954e3ef616526691d52a012e42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f27408dd083eb56f665338cc94361e6b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 2, 5, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_710e774b147504599cb1ca9ccd864d87(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 1024, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2c61eb7bb61552919800f3a4bc6f8aa5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_710e774b147504599cb1ca9ccd864d87
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 1024, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7cba39c23d6cf29119caea16afe70f12(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 4096, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_768beb4386677244dee1e09f8084f087(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cba39c23d6cf29119caea16afe70f12
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 4096, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e731d2e3c529ff0307c78684bca1591(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9874f711042e2bfd107011a52d643e65
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 16384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa5fab6332e1433b0403dc03e2d1b60c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_736429e2d6de83812117a67bd8bf0eec
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 16384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_169bc8da91ad15cf7be39a93e325cde8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c9267bec5ee55f9afae9c6d978b6d09
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 16384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_612087630e0dcc2c72af6775c3d9003f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e8d9b63e2c91b585c5fed809877a683
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 20, 128, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_719b418e1a34bf6366cdcd24fbeb1154(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_892f8c36460cc78452c2435bef822a26
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 40, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fd6c70c47f3c516e2f0f5b346fa410b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23b4cd10e1ef8ee982a07ee0d5b70e73
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 80, 32, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_520d50ac545e6162b5f86a56660e559d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d801a3f63884f9364bf925bb56c70c27
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 12, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0bc02d9fc19e5175109a2be4a3ac0f5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c0688ceb8a47d57e27f99c90a19bb9b
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_543a9a27e8938df4d7737569184f8f3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dff5785c503eefcb9c9b84253e71133
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 49], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a9061a2f676760b282a661fc485a3a55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f8042f2a9c21448974db77e26270663
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 2, 12, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2c67e2b8633c9d2c705cfd02392041d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12174faa73e86c03464ec8a20a68da9e
    def get_inputs(self):
        return [
            paddle.uniform([43, 12, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b0ac7c729582f2369dd5dd70416da9a2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 8, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8ff28b5ea85bf52abca5fd4be54bc7ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b0ac7c729582f2369dd5dd70416da9a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 8, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f16a1270a6c0a5a26bcb0adca1c6522b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 2, 8, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_57fb003ff286328d9a119a5ebebc1e02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f16a1270a6c0a5a26bcb0adca1c6522b
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 2, 8, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_58f468c11d734f8dd9fffb76323389cb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 512, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ea3d993be522b92484df17c64308c475(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58f468c11d734f8dd9fffb76323389cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cea21b67845ee94976c1917716f67ffc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 512, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7e1ef102a5b27f4a7dcf730e5645589d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cea21b67845ee94976c1917716f67ffc
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_004db2b2001a98539fe0d4e3ff5205f8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 176, 176], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7b7040de575c8f1d963cb84edc486a38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_004db2b2001a98539fe0d4e3ff5205f8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 176, 176], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7d33d3baffa3b7acf93705d6a28a01ac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 88, 88], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_92f814e6f38c505c3771e7c0fe2b40c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d33d3baffa3b7acf93705d6a28a01ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 88, 88], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e1f4f5c5018e47ab79513c546f31a2ab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 44, 44], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_507c9f0991d2db8c64ddf3d331e1c4c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1f4f5c5018e47ab79513c546f31a2ab
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d7fb9b71ccc1e783eb5bdb89f855c0ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 22, 22], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9f744a127a31601c180d43970bc48505(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7fb9b71ccc1e783eb5bdb89f855c0ae
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fc5c2475ad006fd7c688fae8ff987215(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 11, 11], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1ae1c347f5a116b01ddf825cfc5ccdcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc5c2475ad006fd7c688fae8ff987215
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4160ece2f83e8707162143468dd7a564(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 176, 176], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1da3590bbd6e14c947fc661344986c6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4160ece2f83e8707162143468dd7a564
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 176, 176], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_696bad3fde869eabc719169e50d980ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 88, 88], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fab388297c8a74b1f0d872dbc7466218(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_696bad3fde869eabc719169e50d980ee
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 88, 88], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5bff4200acbbf93c623f612fe7748cdd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 44, 44], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_909f50056cbc148139a704b5a106723b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bff4200acbbf93c623f612fe7748cdd
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 44, 44], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bf1c5183a962856090970855ec9bc386(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 22, 22], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_11aa192a0c811a58158c81ed2ccfbdf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf1c5183a962856090970855ec9bc386
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 22, 22], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ec846601ee8f8466aa08c85f2871fe02(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 11, 11], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_375cfdaeb597c0cedd858cf210d8e626(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec846601ee8f8466aa08c85f2871fe02
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 11, 11], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7521ee2398b8703704cdeb8d405b74a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e619824f41249824280889dd4e6db702
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_55657fb5204d24fac794f6677ae76fae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7fd087e9b8d337f4e1b6470e74a6e682
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10df1546204e938efab6a68c766e4a9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a65e69570ac7936e6fe370c04384258
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3ec88320ea082b5e15abc7703263bfae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0264b42ee0b22b0800ba3f737b6b7897
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 6, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d08fb43735e1af17a9cbe60dce5f95dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76dee8755d5dd20cf8c34a12ce9eb7c2
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8be14f6c791d87f70b673b3048873f49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60cb79dee0cb1e9fe788d25e98edb957
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 49], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba9b2baeb7b0323e9e76de8c56fcca0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_960222d2177659dd07d97e15a0dc249b
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 2, 6, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f65f263ecaf0bd0a7f650ac3436835c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8718f08430db77a6af404763fce7563f
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0c5724184a7b3f12405d87672c25b77d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 20, 784], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f3af889217f604628e2ed0f3a59f49d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c5724184a7b3f12405d87672c25b77d
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 784], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ec50c8425c07efb66eb45659e5e61cd3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 784], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_66997c947538d59f9d5d09eb0ebf47b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec50c8425c07efb66eb45659e5e61cd3
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 784], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e423dc7ea07b04578889d6ea42f7e74e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 784], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f6d2054b9b033d10a28eceb3b7313688(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e423dc7ea07b04578889d6ea42f7e74e
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 784], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_66997c947538d59f9d5d09eb0ebf47b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec50c8425c07efb66eb45659e5e61cd3
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 784], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc1fdf189ef3e2f628e1d5f54ef08c52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43eede5085a2c790494760fca87050a1
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 196], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8fd0403ce8760c92cb79a69b21de8481(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 1444], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b709cf5ff5d99e5b23653b1983c168f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8fd0403ce8760c92cb79a69b21de8481
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1444], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ec83816c96512b549bbd39ee7ff5a81a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 1444], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bffeeb952fa6ffb773927fd71c896605(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec83816c96512b549bbd39ee7ff5a81a
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 1444], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e4c8c29c2d3a3ab4dfde3d866a9a652f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 116, 32, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5d121ab7f0bc5ea378248a094a5d8ec3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4c8c29c2d3a3ab4dfde3d866a9a652f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 116, 32, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c16e23652ba9807698ac09d1a8404d04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8720a1a95dd8cd9358aef827dc3a5f6
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 12, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0f288fbdc80ad394b57bad99af86f431(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50e179c887d95e472bab06e86b01d414
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d3c5d441fc5f1833890152fc2d40f6a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb9eb20474bb4e4f7d0f85180b7cb4a6
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 49], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_787e37ff7bda6da9deb2e327f11f5265(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07bfc4e5f5542767c9d6eaecc9872d75
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 2, 12, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_69fdf6eae5e56d8d2ca856a3e250f29a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cbc16abd8a3736cc9be4a9e2478f813
    def get_inputs(self):
        return [
            paddle.uniform([11, 12, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_29d53ac7787bbd43eb9622fa7fb4f619(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e1bfc6372783f01f98ed165cbe5693c
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 784], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c5c68140544f0c07cd00234ed62376ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 1764], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b7446285c1cbb3fb0eab14408d8c8232(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5c68140544f0c07cd00234ed62376ee
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1764], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7cd2c5c365f1bc639b05474cb61c5ddf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 1764], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_888c2c34ec88a6ee07fdf16abec4a636(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cd2c5c365f1bc639b05474cb61c5ddf
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 1764], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_98a4dde342bc73ca96d006e07e9af545(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb9f70282899345810179c07997ffef4
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_912910cdc4ece2fb9cb24370b3ae5422(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c2deb8e07905eb4b1bb495ded91337e
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 784], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8f383dbc13ade5f8c112a5e34b8eeecd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 144], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a9e3e0cec9332551316ee485d95fa04a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f383dbc13ade5f8c112a5e34b8eeecd
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 144], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a0aeba2c31530c1bf04644f0cb1fbfef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 144], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_791f658cd19cb9872a3244d21aaa2c06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0aeba2c31530c1bf04644f0cb1fbfef
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 144], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_02b43cb7326c5459a1fed891d504ad9f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 16384, 2, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b6dce9bf1c79aa3f95ed2e4ccf3ab68b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b43cb7326c5459a1fed891d504ad9f
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 2, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0b96d48ee7189cc3b69ead1212cf2812(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 16384, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ae87f24d763e883a88076cfbd78cd664(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b96d48ee7189cc3b69ead1212cf2812
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2abcfddaf7581db1751ef68f75837e9a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_171baf270ba4ca98a4ec6210f03f681e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2abcfddaf7581db1751ef68f75837e9a
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fe8686a820a5d1cfe3af223cfa92a753(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 2, 2, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9537cb60284a7f5a6c02fc509837425d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe8686a820a5d1cfe3af223cfa92a753
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 2, 2, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_73c31545d2b2ed30ce571ccb701a3402(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 1024, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_72f0202c99f677d3c119de4c202184e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73c31545d2b2ed30ce571ccb701a3402
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 1024, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e232df6b5db313370d1004558e343e91(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 16384, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d76fcdb30312f4280cdcb8b7d2ac7a5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e232df6b5db313370d1004558e343e91
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16384, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e528aca4897e67ed3f1cc4caa182d2c7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8192, 2, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9484e3a191cad508082ace11c5336a16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e528aca4897e67ed3f1cc4caa182d2c7
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 2, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4a95e2c578256339b872ec267f1ea19e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8192, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_25ba6898e2f8455e3d1ae84f0373c3bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a95e2c578256339b872ec267f1ea19e
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_52bf600e6a68a77a05f8f66917420ef6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_978eb3f04d7b77bead5085990d1adbec
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d4a63df60863bbb9d3115413151e217c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 2, 2, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a88e0866c740cda3ab53dc0a236c7c01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4a63df60863bbb9d3115413151e217c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 2, 2, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f4ef289203f2e5a7975ba007aeb6dddf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 512, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2443bef4e5c16c686806aa6662bc5793(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4ef289203f2e5a7975ba007aeb6dddf
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 512, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_67c2cfdf126d367a27bef711597da50a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 8192, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_65b9f7797bca8579a8fb3e994adab39a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67c2cfdf126d367a27bef711597da50a
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8192, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c086dd764d41381070e45945c7e7225b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 184, 280], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ca4e609e99f1bbcdc86c93a43c6bef19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c086dd764d41381070e45945c7e7225b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 184, 280], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_542218b6136c5e14ea5a11d78892e6c2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 92, 140], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7df3b9d2c89b7949286ff8ddb5a0d3cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_542218b6136c5e14ea5a11d78892e6c2
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7eea9697ca6c6ee8807b2bc184f4fd0f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 46, 70], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0193207d3a87c336983588a4cd63186b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eea9697ca6c6ee8807b2bc184f4fd0f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_42615672f64c2a7764af169f86f0acd7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 23, 35], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_044b40271d0713504bffff3e1c33f003(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42615672f64c2a7764af169f86f0acd7
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b25ecc1272b26e9dd05c968de9744613(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 12, 18], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8ec0d594f6566d492bb06e9539f14d1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b25ecc1272b26e9dd05c968de9744613
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fd3e66aaec9a8129a6ae26e440b2fb11(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 184, 280], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5c6cf26eba540931ba0d4e9c7eb56256(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd3e66aaec9a8129a6ae26e440b2fb11
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 184, 280], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d84aef348ae946ff65c2d60b916b1554(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 92, 140], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e23c3c434f5e3d835ed9312c76b49f0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d84aef348ae946ff65c2d60b916b1554
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7c8516c04338d4c648c7d9a1c2c25797(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 46, 70], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_00e71fb8023ea57111192ccaf8e86c27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c8516c04338d4c648c7d9a1c2c25797
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_78f99e551210d2c7a338c837aeff36fe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 23, 35], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6e753f9d82d0e1c25b95e438c1daa539(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78f99e551210d2c7a338c837aeff36fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3b7304a101a8de5f7ec89f91f44c1006(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 12, 18], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_11c53c773668f588d8ba557aab598af8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b7304a101a8de5f7ec89f91f44c1006
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0c1aa723bd54130f474e3a66938d139d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 16, 64, 320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3a8185ea02c40a981833a5d669090fb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c1aa723bd54130f474e3a66938d139d
    def get_inputs(self):
        return [
            paddle.uniform([4, 16, 64, 320], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a92d078e7f2b5979df029bb491e7a692(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 512, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0674a88374671bbfac1ef9176d220503(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a92d078e7f2b5979df029bb491e7a692
    def get_inputs(self):
        return [
            paddle.uniform([4, 512, 256], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_333f54d70101912ab921726d45fb4758(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 80, 144], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8e644378cbf487a0bf600bb4566ade55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_333f54d70101912ab921726d45fb4758
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 80, 144], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_55fb65de185b959fb46a620ffe5d5763(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 197, 3, 3, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6eef4225fb1ccb4bbdebdd7b6c424d57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55fb65de185b959fb46a620ffe5d5763
    def get_inputs(self):
        return [
            paddle.uniform([86, 197, 3, 3, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2d2b3f418ff49b93d9443573a8df6007(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 3, 197, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f9065ca767f2750c5ceaa963252c3064(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d2b3f418ff49b93d9443573a8df6007
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 197, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ad9fb1d3139427203d613be91e0d5763(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 3, 197, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4e48b898a6fc8dfc58c90a988c184371(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad9fb1d3139427203d613be91e0d5763
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 197, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f1841d7dc751495753d17b845ebf005d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32768, 1, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_08bfb8a37bb59df77d087cda1210e6fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1841d7dc751495753d17b845ebf005d
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 1, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f03b015a031dbd8a5f1ce136ffd39747(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32768, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2d8775c36b47c34203a010f4669599ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f03b015a031dbd8a5f1ce136ffd39747
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_306658df373931a5a109d6eaeb891608(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6dcf1c6843d16a100e6de5038bef0ace(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_306658df373931a5a109d6eaeb891608
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_84c756b536e05f75680b37715c0f731d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 2, 1, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dce90ae196ad2dff27d42336cd7c712d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84c756b536e05f75680b37715c0f731d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 2, 1, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_281f41aaab29f3ed12869f1b207b9d1d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 512, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f2438b0918d8f51f6d179dbd4ef94c9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_281f41aaab29f3ed12869f1b207b9d1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 512, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a5d0cfaaf3c1dfa473013e302af8bcac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 32768, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a5f9f3cbc952fd2075c95264e6751d15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5d0cfaaf3c1dfa473013e302af8bcac
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32768, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_01569d6d711e090c69926ab2c745c449(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 4096], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e6a7314ff4eee2c6caeeef9ab1258e64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01569d6d711e090c69926ab2c745c449
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 4096], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0b4cb7f4e82588b511e7c22a4299938a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 8, 64, 320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_98c8c01deb9f1546b7b7bd3a36026628(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b4cb7f4e82588b511e7c22a4299938a
    def get_inputs(self):
        return [
            paddle.uniform([4, 8, 64, 320], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8c84256526bcc9d36949fbb569c5b0bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 512, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ef0dc26dbf72f5f03a58d121381e138e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c84256526bcc9d36949fbb569c5b0bf
    def get_inputs(self):
        return [
            paddle.uniform([4, 512, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d2bb59b7f4a71d24aa04e65dafd9223c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6e0ee1ac9852d27bdf908e9f8e9e94c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2bb59b7f4a71d24aa04e65dafd9223c
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1a37be9ee1195b7c221974521c0531a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3024, 4, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f19837e7d555d48f3dccbe87e7bf1307(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a37be9ee1195b7c221974521c0531a6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_484c238e6df29f547f33f268e9fb6ff3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 196, 4, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aeb22f6577cf170a6e76b2a8b1d9fcb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_484c238e6df29f547f33f268e9fb6ff3
    def get_inputs(self):
        return [
            paddle.uniform([22, 196, 4, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aeb22f6577cf170a6e76b2a8b1d9fcb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_484c238e6df29f547f33f268e9fb6ff3
    def get_inputs(self):
        return [
            paddle.uniform([22, 196, 4, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_35f38826c3b6c1a7e66cdd0e52b9ee65(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 196, 4, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e19f64a54a12598f046c4c2806ff2618(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35f38826c3b6c1a7e66cdd0e52b9ee65
    def get_inputs(self):
        return [
            paddle.uniform([22, 196, 4, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c9b966bedff7c3269c23d09cfb084f0b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 4, 196, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_955a94c2a6e04a90adf48b077ec5181f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9b966bedff7c3269c23d09cfb084f0b
    def get_inputs(self):
        return [
            paddle.uniform([22, 4, 196, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_18b8d9a39fb893fb184e100b59404a17(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[1, 0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 196], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_544c1c7f09e905d28ac95414cb39fdac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_18b8d9a39fb893fb184e100b59404a17
    def get_inputs(self):
        return [
            paddle.uniform([4, 196], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_647082ef376f5a73c0964f696d448383(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[1, 0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[38416, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3efa7a09d1db980e34eefdeb8e5173af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_647082ef376f5a73c0964f696d448383
    def get_inputs(self):
        return [
            paddle.uniform([38416, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8a6a07b9d8c2ae2d9b08dcd058d91dc9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 192, 288], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fa67dca7c9cfd32e1abf2fcbf417bf2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a6a07b9d8c2ae2d9b08dcd058d91dc9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 192, 288], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2c81a6e6581cc06cfe364cffc890013f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 96, 144], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e312409c08cf0b6d54c1628bb23217f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c81a6e6581cc06cfe364cffc890013f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e4e288a939db1e09e043df3c848391a2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 48, 72], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_30659222af492f698275fdb0d52e82de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4e288a939db1e09e043df3c848391a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_42e71324cdeb85098b19c49740a479b6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 24, 36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_638e9bb288da87c68f0987a83f0bd48a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42e71324cdeb85098b19c49740a479b6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8ec0d594f6566d492bb06e9539f14d1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b25ecc1272b26e9dd05c968de9744613
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_88fe47fec67a58d61a4a3d1d16cc3328(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 192, 288], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fe8da55f33753afe5e25a2de998af80f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88fe47fec67a58d61a4a3d1d16cc3328
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 192, 288], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3c598154bd658680190eba6eeb28f153(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 96, 144], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3bb0f7a7afec015af491deb9bf1b1dbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c598154bd658680190eba6eeb28f153
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_637afb49112ac4682401741d34792b82(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 48, 72], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_406227e872687fb4fbe361105369427d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_637afb49112ac4682401741d34792b82
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6bf8b2f21ca5a060d6a86147919864d6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 24, 36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7c5358ef78698883f35f256243ad720b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bf8b2f21ca5a060d6a86147919864d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_11c53c773668f588d8ba557aab598af8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b7304a101a8de5f7ec89f91f44c1006
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_90ca448d35f938a13e0e478d14020e43(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 160, 3, 8, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_12bf76e4f73044179c7bd8f9668f2a56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90ca448d35f938a13e0e478d14020e43
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 3, 8, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_761b85a9689b58fa781b6ec54cd18a95(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 8, 160, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ad28d7fd6e8c45724a0a0b6aa83f8d56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_761b85a9689b58fa781b6ec54cd18a95
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 160, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cda03d70089208e085bddc13789dc777(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 8, 160, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6aa62078223ab8ccb57b71a72e85d0e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cda03d70089208e085bddc13789dc777
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 160, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_993b6f91c7c33514cbd5994bb0bdf580(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5873366485593aa1e6a050d8ea9217b3
    def get_inputs(self):
        return [
            paddle.uniform([22, 49, 8, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0279ba42dd029a85640b4f3c3b71e93c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[1, 0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8, 196], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_73b0bc9ad3e3452ed414f4b7e9553840(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0279ba42dd029a85640b4f3c3b71e93c
    def get_inputs(self):
        return [
            paddle.uniform([8, 196], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2f60a274c4db64220fd4a2093b0f04c2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[1, 0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[9604, 8], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d68222085b4f280df56ef124d1b1dd7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f60a274c4db64220fd4a2093b0f04c2
    def get_inputs(self):
        return [
            paddle.uniform([9604, 8], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_71acc1dd4e0b1287d85016dbfd1dd6b5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 8, 196, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_873120d829582701c28b8698b85c3e78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71acc1dd4e0b1287d85016dbfd1dd6b5
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 196, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c50cd402ccb228b05ad43e6e5e81bd3c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 16, 12, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_50cf77a3c2ad6137940202a828afdd48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c50cd402ccb228b05ad43e6e5e81bd3c
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 12, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_50cf77a3c2ad6137940202a828afdd48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c50cd402ccb228b05ad43e6e5e81bd3c
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 12, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9c697a05358bea34b58d24b470a85fda(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 16, 12, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b03507bef5ac5bf1bd821f6fc76bec03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c697a05358bea34b58d24b470a85fda
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 12, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d8e9f28a4ce34fd76baf94bb2d1d324c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 12, 16, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7d8b9b075176e8b8aa0ff8c9ab02c734(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8e9f28a4ce34fd76baf94bb2d1d324c
    def get_inputs(self):
        return [
            paddle.uniform([22, 12, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6ecfc7e48c9f90e0850385bce9fcaf98(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[1, 0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[12, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f4daa8707153561968a8edea8de8c04f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ecfc7e48c9f90e0850385bce9fcaf98
    def get_inputs(self):
        return [
            paddle.uniform([12, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_db036769e2ad1e7d56fd5962db5037b3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[1, 0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[256, 12], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_445bd6afcff9b8ccece529058657f62b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db036769e2ad1e7d56fd5962db5037b3
    def get_inputs(self):
        return [
            paddle.uniform([256, 12], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fef519db7a9cf4aaf76fba81a282032e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1174, 3, 6, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0cda5e88804d07031b39fbdcda592913(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fef519db7a9cf4aaf76fba81a282032e
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 3, 6, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a548a2aa221ad914f7023dc74380aa27(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 1174, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_51aea8b453202735739e04f3aed9fcc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a548a2aa221ad914f7023dc74380aa27
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1174, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f9e54586d17ae3b643e97d6f3e428c8a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 1174, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_099b821861d8aa0a76f11e65ef3f3e59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9e54586d17ae3b643e97d6f3e428c8a
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1174, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_23539525be76bcef05dda785424eda3c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 150, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_88c79000928dd132e68c1087c6987cee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23539525be76bcef05dda785424eda3c
    def get_inputs(self):
        return [
            paddle.uniform([1, 150, 256], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d51ac0a1c14cf645d9feb46ce0f6512b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 16, 32, 160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_83498c916f602cf6d490b5e7b8c89b4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d51ac0a1c14cf645d9feb46ce0f6512b
    def get_inputs(self):
        return [
            paddle.uniform([4, 16, 32, 160], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_07d1e12dc263efa9acca3814d2dea40f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 256, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3baf4f185a7a4115dae50911c2fef09a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07d1e12dc263efa9acca3814d2dea40f
    def get_inputs(self):
        return [
            paddle.uniform([4, 256, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7e98c6e3c5948eff2f4fac5ee115f6b4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 58, 64, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_69217a2728b226ef26723bc20e45bf71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e98c6e3c5948eff2f4fac5ee115f6b4
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 58, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_96ce4c0777c0749d36605d7ffb3b65ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfb86674f08e2524dd57ca19edcc36a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 4624], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c46f09b3c8bd6dcb324d3a4807d46066(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 4624], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3be42d6e433bb1703ff000bf2e74109b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c46f09b3c8bd6dcb324d3a4807d46066
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 4624], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_645fb0c2c28c847a8530df8cb42db822(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_112b2548816103019986f69677d7b728
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 2304], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3c04ce9e038dd692713a55140cd80211(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 2304], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f305094a423dc614c5dce327023e32db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c04ce9e038dd692713a55140cd80211
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 2304], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f70dc5608adc55511e8e76930a187822(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 2304], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b094cc7cc45b8698b48bb61834e333c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f70dc5608adc55511e8e76930a187822
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2304], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_db168f22dc31d798cdc32e7ceee1c9c8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1174, 3, 12, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_512e29d34a3f5b2f90da00e28c3088dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db168f22dc31d798cdc32e7ceee1c9c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 3, 12, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_01aaa29f674ebfe6cd972d52bc4e034e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 1174, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dc14770944a1205c38f9f255f2d9448d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01aaa29f674ebfe6cd972d52bc4e034e
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1174, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_32af6bd3107bb7be8474801d88c7ec81(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 1174, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_248f79409adfa5ad64f28776d4dc30d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32af6bd3107bb7be8474801d88c7ec81
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1174, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3771127f24351c4e728ff51c2cb7ed25(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 65536, 1, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_570a6b630697c59d3f1f1531e15b82e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3771127f24351c4e728ff51c2cb7ed25
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 1, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7f2a362cb7e6793ab263d0f914ef7bcd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 65536, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e21aab9df4795642a78d4a7b97aba31c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f2a362cb7e6793ab263d0f914ef7bcd
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_171baf270ba4ca98a4ec6210f03f681e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2abcfddaf7581db1751ef68f75837e9a
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fb2c4b7acc55c9bda5acfa5aa5083468(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 2, 1, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f427526ae7bdbd69c0e6733c92a2ba5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb2c4b7acc55c9bda5acfa5aa5083468
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 2, 1, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8363c5b5a0b47b74a4c2cc63d5e74f9c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 1024, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_af09a51f17f4a3a1ad5835414cc9b325(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8363c5b5a0b47b74a4c2cc63d5e74f9c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 1024, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5bb93f82dc6badabcd84d4df755fc1f6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 65536, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d8dd2704b293831d90d287e8952b84f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb93f82dc6badabcd84d4df755fc1f6
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 65536, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4d8c425c849a5c4be4b138deeed99498(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_407e86043743db80b886354782c8f9d0
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 3136], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_188efac39328cf8b7fceba4e5dc2464b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 50, 3, 8, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0765d08d3aef0957168bd35915345ff0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_188efac39328cf8b7fceba4e5dc2464b
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 3, 8, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7ed745f7f86f6dd1c5a7236ed555b0d5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 8, 50, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2460c8e7a09ba81f1f781ec948f19faa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ed745f7f86f6dd1c5a7236ed555b0d5
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 50, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_116276ba383e3639512ce3b72a1c4dc8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 8, 50, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8bdf4e0ab6d1585e55cdebdf60070c7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_116276ba383e3639512ce3b72a1c4dc8
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 50, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fdf1de3b8bec711d5ea1da6362107f15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf4a2b21b5d3cf8f17e37d73fa57ab96
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_176e9358909137c69194ec61f8bb4733(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51f6e4e493661b6052e041f3ff5dbbbc
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 3136], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e1b43b443c67d632a75e36f51d7ec694(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 49, 16, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_238bc55abd630b789e92e585a4759659(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1b43b443c67d632a75e36f51d7ec694
    def get_inputs(self):
        return [
            paddle.uniform([22, 49, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_52ff648e9dd84f5c44548e4bd75fd23d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 49, 16, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e2e02766b42277159a50146a494878bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_52ff648e9dd84f5c44548e4bd75fd23d
    def get_inputs(self):
        return [
            paddle.uniform([22, 49, 16, 64], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()