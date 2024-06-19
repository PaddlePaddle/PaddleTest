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



class PrimitiveOp_837cb427141c8a03b2f90eb13eaec78a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 23, 23, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9e356f473363ca3c769689f670dfc91f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_837cb427141c8a03b2f90eb13eaec78a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_6a71db394be806519d270b2ee321db04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_380c72d3b13276b5deabe54ea6141818
    def get_inputs(self):
        return [
            paddle.uniform([100, 2, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e8ae90989382d44f291abe77cce4a86f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[15200, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_493de3abb47b87f3f0185abb6f8e031f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8ae90989382d44f291abe77cce4a86f
    def get_inputs(self):
        return [
            paddle.uniform([15200, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_29f563030f4e643891e72435ef5119f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1542, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_16de1073bf0c8d9a37c7cb311cbb1037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29f563030f4e643891e72435ef5119f9
    def get_inputs(self):
        return [
            paddle.uniform([1542, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_86c72728617af640227a9ce9e1c7209f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2361, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9138f8f6197c8689aa7cae8faed55949(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86c72728617af640227a9ce9e1c7209f
    def get_inputs(self):
        return [
            paddle.uniform([2361, 4], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_1c889adb4c0087c1a3d28250d78b19a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b86938a5e248737c86607aed172a9aaa
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_6848f47761d20600e2c540f2a7c69346(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_160239381517b9853a36bd639404d4c4
    def get_inputs(self):
        return [
            paddle.uniform([47, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3478fb9827a292eaad9786f45975aad7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 76, 76, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8a77f12278b6e195021bc82ce3d8b9f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3478fb9827a292eaad9786f45975aad7
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_09c7bb3094ed5885d06dddb6bd05db77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb58e13e7c749ce50045faa688f10d8d
    def get_inputs(self):
        return [
            paddle.uniform([300, 2, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_787b17b6b35905e81ac4851920b9c752(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3800, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_af1f761894e7bb255856487fc26f5ac7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_787b17b6b35905e81ac4851920b9c752
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_af1f761894e7bb255856487fc26f5ac7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_787b17b6b35905e81ac4851920b9c752
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_187bdeb6943fd3e18f2775d9d079ea11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_694545f2b7efe73e654c98068614e16e
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_2facb00784e01e1af35940196f2f4b67(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2204, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_946b8b04a5923f9c9449eab19ccafefa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2facb00784e01e1af35940196f2f4b67
    def get_inputs(self):
        return [
            paddle.uniform([2204, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e3babd1282df99565387c3d5d751b83c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2053, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_40ed90fccfdee7a255e7df46d00d64fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3babd1282df99565387c3d5d751b83c
    def get_inputs(self):
        return [
            paddle.uniform([2053, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_8dd8d18fafc6b096627ab0a6b2b95d0d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 21, 21, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4bf7c260f55098f95d39dc1fb9d2b389(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dd8d18fafc6b096627ab0a6b2b95d0d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_86e2402a705e0a0e5c0dd81caf189a82(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fecaf8683a2c89e19ad8afa1e9820d7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86e2402a705e0a0e5c0dd81caf189a82
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_db90d0d5d4ecaafd906cadb2ca10dd40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0dcb72f58c1bf0f6727269af285d7d73
    def get_inputs(self):
        return [
            paddle.uniform([103, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_fb98826417fe973c3ab7befceb9816d2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[950, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9613c1a3b58e63ac3a4e71098979b8e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb98826417fe973c3ab7befceb9816d2
    def get_inputs(self):
        return [
            paddle.uniform([950, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9fdf2c2b4a5b329d67200c895807c0cc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 84, 84, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f14f95039bebd4ff5cc8852c32ec2279(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9fdf2c2b4a5b329d67200c895807c0cc
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_583fc7ad747dbfe6781071dc7d9d2c9a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1825, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b63236169fd7a7d43e5f36e7eae127ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_583fc7ad747dbfe6781071dc7d9d2c9a
    def get_inputs(self):
        return [
            paddle.uniform([1825, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9ebc6e9cd17c649fbae436f96d9924a0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4832], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c30764cd2610977c75e60c535b9c6105(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ebc6e9cd17c649fbae436f96d9924a0
    def get_inputs(self):
        return [
            paddle.uniform([4832], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_64d78ba547fbbda306b3d274b0f2d280(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3087, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_071e97dfe008deb7c2a41f2866e31cf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64d78ba547fbbda306b3d274b0f2d280
    def get_inputs(self):
        return [
            paddle.uniform([3087, 4], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_1ba8cfc23657f61922a9bd84346194e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8155fec171bea0b2c7f2d039e4b9a472
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_8dbaaf4555aad840f05a5766f394c43f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8816, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7e334087447825851a30d1d2f6d70098(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dbaaf4555aad840f05a5766f394c43f
    def get_inputs(self):
        return [
            paddle.uniform([8816, 80], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_a9fd69528e17d89b09bc395b17d9bee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99d93a50df9d281b22231092485fa0da
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_87906955c33cf51ccde2061dca68526b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f661a5141cec2d4495e1f81d65e5b436
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e0488f685d0ced6a255a5c8a22456162(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2119, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_116ce9beaeb003b868bb0cc172ae93e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0488f685d0ced6a255a5c8a22456162
    def get_inputs(self):
        return [
            paddle.uniform([2119, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_638d3358f7713cb3cf0e3864917b8c29(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 42, 42, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_84a523d9ed5088532d854da634bf39d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_638d3358f7713cb3cf0e3864917b8c29
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_4135cd31fde26720aba241a7243aaa16(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 92, 92, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f1337cd918d38b01ad3f7e92fd3b263f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4135cd31fde26720aba241a7243aaa16
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d9b064a2366fdbd4ef8e29baae81b8bd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[150, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a6a252f4eeddf2912cd2c4cec15c2d61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b064a2366fdbd4ef8e29baae81b8bd
    def get_inputs(self):
        return [
            paddle.uniform([150, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_2797ea69ee4cc3ba0da5254dcbd2e3c5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[70, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_20f0bba5548757ea369ecf97888fffdf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2797ea69ee4cc3ba0da5254dcbd2e3c5
    def get_inputs(self):
        return [
            paddle.uniform([70, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_645784c373529ccac989b0eed57c6038(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 44, 44, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4cb46bf2d5f428872abdac78e8fbbfbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_645784c373529ccac989b0eed57c6038
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e3bf12c1e7a1e48a4013c7981cf1dee4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 11, 11, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bca56cf767311b047018316968eaa530(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3bf12c1e7a1e48a4013c7981cf1dee4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_a858b574322d79ba0243c7d67786f47a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dfe8a64cb50bfe2bcb3b76550a033fe
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_bed36877b9074e68f05e0f4a117d4cf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_290f3d8b3a3e03a240af73601481ebe9
    def get_inputs(self):
        return [
            paddle.uniform([56, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_391e9f30db2849017acc18ed7c74019b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5606, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_99b4dda5ec1fa2e6589e2e8eb17fd3c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_391e9f30db2849017acc18ed7c74019b
    def get_inputs(self):
        return [
            paddle.uniform([5606, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f952ee5fdff37335f2534f2434df1357(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 19, 19, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f1a220a46e35d4d0da011db842427e7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f952ee5fdff37335f2534f2434df1357
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b0c30619930a79d199f888fe4525492c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1036, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8f142d22cfaa280176cf997da61b3c56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b0c30619930a79d199f888fe4525492c
    def get_inputs(self):
        return [
            paddle.uniform([1036, 4], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_8fe2830bb49e3928c192b798a6d32a5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5da4620298afe229315d5b1aabb58d5c
    def get_inputs(self):
        return [
            paddle.uniform([53, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ce9b45339206f674620d10d638e6903f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1809, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_57f956033b5945a63f5736d7cc8f91bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce9b45339206f674620d10d638e6903f
    def get_inputs(self):
        return [
            paddle.uniform([1809, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fecaf8683a2c89e19ad8afa1e9820d7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86e2402a705e0a0e5c0dd81caf189a82
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_9eb120fb55a0b659156f8b173fa63482(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a7dc964ab4827838ef16c98d448eeb2
    def get_inputs(self):
        return [
            paddle.uniform([84, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d554395fc6a598af9c87f795165d6ac8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 12, 12, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2ee0c3233cd85c05d94b9723b6441dd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d554395fc6a598af9c87f795165d6ac8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9dafdfe52e5e1113587d5343ba14d578(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[17421], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b2728816c0ec48d7d335ee95af76f360(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9dafdfe52e5e1113587d5343ba14d578
    def get_inputs(self):
        return [
            paddle.uniform([17421], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_057561c881893fd490d54f463e7542a2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 46, 46, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2ff908f6e1a76060417427497f4a65d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_057561c881893fd490d54f463e7542a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_e9a6d6fd6a75c230190e90e10e01ce8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a61b537fdd5da6e3871c0805e59ae14
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_9ed509e419c1431da292b0ecdcb5b6f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7dfb0a11cf57c66c218b7433152ec835
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c9a3fc18e45a3ed8e95a53c61819602e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 48, 48, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_49232ad1853d9bc5d8ba79d34a14a894(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9a3fc18e45a3ed8e95a53c61819602e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_233dbafe4ba4b4e0ac467b6112f9a21e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 24, 24, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b74784df5fd79aef12809ed6c3436473(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_233dbafe4ba4b4e0ac467b6112f9a21e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_cd806c550963b9f8b89fd0a0b03fce3e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4179, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_97097605522835aead9830888ece2f93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd806c550963b9f8b89fd0a0b03fce3e
    def get_inputs(self):
        return [
            paddle.uniform([4179, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_487c6208529eb73993f5fa47640e69df(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 38, 38, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_41792dcc0a9963e29cd5c5d8b832bc86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487c6208529eb73993f5fa47640e69df
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_77958d28a93dd6fedaf90e80c7180d92(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[551, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ab6fce11e9101d4b5eea763870eea957(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77958d28a93dd6fedaf90e80c7180d92
    def get_inputs(self):
        return [
            paddle.uniform([551, 80], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_078af3764e861a97997bfaeb99f12fa3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e42c7490339aee257e47ac0070cf768
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_220305de4100e9fc1601b7c275cfe766(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b41f0c7a4755f75643a6fbf31fcb7605
    def get_inputs(self):
        return [
            paddle.uniform([52, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_db39f88185ff33dbb7161e521799425e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4662, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0d4c1d1c5eb455adeb0e91dc85353a14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db39f88185ff33dbb7161e521799425e
    def get_inputs(self):
        return [
            paddle.uniform([4662, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_74627fa413806566e3a6fd3292b1eb70(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3857, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d4d7399ed21fa9e70d3cab85db3c4010(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74627fa413806566e3a6fd3292b1eb70
    def get_inputs(self):
        return [
            paddle.uniform([3857, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_12df8d7c40e7d66415a796f390cf949e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 22, 22, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0d3b8f1fbdf61e32d2f1e01b9a1abc81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12df8d7c40e7d66415a796f390cf949e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


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


class TestPrimitiveOp_36a0116c9fa866bc1f07c416c03ac458(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2e1a7c83c21c85aa6cf7cd533ff36614
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f51e5d9443655f38e11d2bf596f5df71(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[247, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ade3002e4091b7ae0106f797a412baca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f51e5d9443655f38e11d2bf596f5df71
    def get_inputs(self):
        return [
            paddle.uniform([247, 80], dtype='float32', min=-0.5, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()