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



class PrimitiveOp_39dbb052161527da1935b24c576bc7a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 72, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ae3de7132487da111a588cd81f930a1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39dbb052161527da1935b24c576bc7a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_411861d6a11339a9670f717d2e2a0999(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 92, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dbc20eb3d47cdf49defb780bdf55617d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_411861d6a11339a9670f717d2e2a0999
    def get_inputs(self):
        return [
            paddle.uniform([1, 92, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_f4ce8c145bb1ee2812711677ef282621(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 2048, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c02acd2dc48c1965ac0340d5d3e18358(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4ce8c145bb1ee2812711677ef282621
    def get_inputs(self):
        return [
            paddle.uniform([22, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_760aeeb0d4df3f35457e3e306fc61929(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 960, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2f94ed777c78a6e1270e7308e6ceb1c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_760aeeb0d4df3f35457e3e306fc61929
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_0b3571b39887b3b7f88714732ef1f113(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 480, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_009820fbde8c0bb5e2eb539564c3d49a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b3571b39887b3b7f88714732ef1f113
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_74c273d9364ec687b97bcb633bea7dab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 336, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8c5df5db6bb8c99491c0cc85fc6c8ae0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74c273d9364ec687b97bcb633bea7dab
    def get_inputs(self):
        return [
            paddle.uniform([10, 336, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_c12843e6db53b1573878fc36d6ee1a24(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 3549, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eeb81d929b9f7207632a58a620a7a34b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c12843e6db53b1573878fc36d6ee1a24
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 3549, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_8ad512f365c4bba2a479e36bf685cb60(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 60, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d4af50a5a1c6fb1949e86795cec3d158(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ad512f365c4bba2a479e36bf685cb60
    def get_inputs(self):
        return [
            paddle.uniform([10, 60, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_13e304be87067a63d9b39c3b79a741f2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3800, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d314a816fb6797f3c44fe3cf8061faef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13e304be87067a63d9b39c3b79a741f2
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3800, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_157384c53426bda7db3a8bbd38579cb7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[150, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ca0d5908fe680126c50362e2384fa1cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_157384c53426bda7db3a8bbd38579cb7
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[150, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_7e54e63edd52a9e50942d3791518af60(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 336, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6a0fc12a83ccfb1f8a4a266eaced6173(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e54e63edd52a9e50942d3791518af60
    def get_inputs(self):
        return [
            paddle.uniform([145, 336, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6a0fc12a83ccfb1f8a4a266eaced6173(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e54e63edd52a9e50942d3791518af60
    def get_inputs(self):
        return [
            paddle.uniform([145, 336, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_e8ad967129eafe360518c0b04dc4a359(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[40, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_74b2977f44a175889ae1c8f4c87a0a50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8ad967129eafe360518c0b04dc4a359
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[40, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d314a816fb6797f3c44fe3cf8061faef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13e304be87067a63d9b39c3b79a741f2
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3800, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_61d665c3f70278e99fc55728fda9dd65(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c78fa3bc9080eb68b6f57fb20991e026(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61d665c3f70278e99fc55728fda9dd65
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.213313579559326], [1.9694278240203857], [2.2759838104248047], [1.9948484897613525], [2.2433714866638184], [2.026952028274536], [2.2625842094421387], [2.3940014839172363], [2.1893672943115234], [1.9999949932098389], [2.039778232574463], [2.0384669303894043], [2.334479570388794], [1.9468075037002563], [1.8638880252838135], [2.090146541595459]], dtype='float32').reshape([16, 1]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_108379dc5034675bf26df32d49380bd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61d665c3f70278e99fc55728fda9dd65
    def get_inputs(self):
        return [
            paddle.to_tensor([[1.9784406423568726], [1.894113302230835], [1.8887768983840942], [1.9572476148605347], [1.8944146633148193], [2.0069973468780518], [2.022692918777466], [2.200212001800537], [1.8431483507156372], [1.8886048793792725], [2.1080193519592285], [2.2595934867858887], [2.2477219104766846], [2.0040104389190674], [2.1124138832092285], [2.153306245803833]], dtype='float32').reshape([16, 1]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_b504e42e3274c82f8bbfa269cb13e23a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 240, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0f0420243059c482bd0f562f38ccda99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b504e42e3274c82f8bbfa269cb13e23a
    def get_inputs(self):
        return [
            paddle.uniform([145, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_653f15d54be6ecdded16910ef762163b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 7581, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_92748aeaac39959e083a493f9b8d953b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_653f15d54be6ecdded16910ef762163b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 7581, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_8d67e9e37843e8de1c007f5cf3c90fbc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 18, 64, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_11f7fa4465228c14699e4293665b1e58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d67e9e37843e8de1c007f5cf3c90fbc
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 18, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_11f7fa4465228c14699e4293665b1e58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d67e9e37843e8de1c007f5cf3c90fbc
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 18, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_b400937e6a191531310b8f98d8992100(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 1, 66, 130], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_286489b5eeb8c1cb47409772b0cb5f2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b400937e6a191531310b8f98d8992100
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 66, 130], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4d5dedfe2c3924054227e8d61c238e61(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 4725, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5a800035aef51bdd7c57664ecc95086e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d5dedfe2c3924054227e8d61c238e61
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4725, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d6ac23f2da1715824bcb47cfe37dd29b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 60, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aef67eabfbc2bb3ab50bbee19d523179(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6ac23f2da1715824bcb47cfe37dd29b
    def get_inputs(self):
        return [
            paddle.uniform([22, 60, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_c16979f803ad9cfb857a8b9ca142747a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 872, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c9c49846614229d1c2568dccd8e10757(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c16979f803ad9cfb857a8b9ca142747a
    def get_inputs(self):
        return [
            paddle.uniform([1, 872, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_a6003985d38dfad5db4c10e94c2003d4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1758, 4, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_40e09cf2ba6fa86d2b21c34e134be4ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6003985d38dfad5db4c10e94c2003d4
    def get_inputs(self):
        return [
            paddle.uniform([1758, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_40e09cf2ba6fa86d2b21c34e134be4ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6003985d38dfad5db4c10e94c2003d4
    def get_inputs(self):
        return [
            paddle.uniform([1758, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_7d51307feb4bf9e593f34138f60c918d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 8400, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0b74c30ec7ec562be756e2af18bafc5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d51307feb4bf9e593f34138f60c918d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8400, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d4d43bcbbd77d19bf34e98d9d221d34d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 336, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2fc907fe8015667ac84f244a953c12a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d43bcbbd77d19bf34e98d9d221d34d
    def get_inputs(self):
        return [
            paddle.uniform([171, 336, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_d7b88f96ee2882642a785a7089eccbd1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 768, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_996164280700c300ad7bbabad714f10c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7b88f96ee2882642a785a7089eccbd1
    def get_inputs(self):
        return [
            paddle.uniform([43, 768, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_eeb81d929b9f7207632a58a620a7a34b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c12843e6db53b1573878fc36d6ee1a24
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 3549, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_9fdf548e509963ffd599a6ae3beac2de(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 240, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5b376c2a8a2ef479c6b7ea607b4604c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9fdf548e509963ffd599a6ae3beac2de
    def get_inputs(self):
        return [
            paddle.uniform([10, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_e3ff30efe9fcc5fc9e23260108f2c148(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5593, 4, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d3819805ffb17eb30c192c61b4654fe1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3ff30efe9fcc5fc9e23260108f2c148
    def get_inputs(self):
        return [
            paddle.uniform([5593, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d3819805ffb17eb30c192c61b4654fe1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3ff30efe9fcc5fc9e23260108f2c148
    def get_inputs(self):
        return [
            paddle.uniform([5593, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_9be0e467d177e090078581bc4b52d17e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[36, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_582fae4177000a67f799763422f82249(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9be0e467d177e090078581bc4b52d17e
    def get_inputs(self):
        return [
            paddle.uniform([36, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_582fae4177000a67f799763422f82249(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9be0e467d177e090078581bc4b52d17e
    def get_inputs(self):
        return [
            paddle.uniform([36, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_42e40bc296ece1c4dc57790d0a8980c7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 1000, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_97b0768fd8ea093bc05acebc3f4dd71e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42e40bc296ece1c4dc57790d0a8980c7
    def get_inputs(self):
        return [
            paddle.uniform([43, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8c5df5db6bb8c99491c0cc85fc6c8ae0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74c273d9364ec687b97bcb633bea7dab
    def get_inputs(self):
        return [
            paddle.uniform([10, 336, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_c126e635a7603e82955a8d7217bb3445(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[15200, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7dedfd97566d5e518ab13ed5a10c44af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c126e635a7603e82955a8d7217bb3445
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[15200, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7dedfd97566d5e518ab13ed5a10c44af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c126e635a7603e82955a8d7217bb3445
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[15200, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6470ef7e42c26180e225df41432f5007(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 36, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0d59b5e6cc90491cd3849fb0916ac78b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6470ef7e42c26180e225df41432f5007
    def get_inputs(self):
        return [
            paddle.uniform([10, 36, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_a06b4ff182bd2cfd23ecb9b114f3516c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 1280, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_20f5192dd2d50722d5c78e00a361da7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a06b4ff182bd2cfd23ecb9b114f3516c
    def get_inputs(self):
        return [
            paddle.uniform([43, 1280, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_2e0bcab0e8552c9bab3f97735dc2daf2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 1000, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0ca0f32e9ed8fb2c7659e0107a2077c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2e0bcab0e8552c9bab3f97735dc2daf2
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_0d5b8d3666e23513a759343cfadcbb7b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 480, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_95e768ce229b4df4c945819398a921c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d5b8d3666e23513a759343cfadcbb7b
    def get_inputs(self):
        return [
            paddle.uniform([10, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_a43f2bb9a9a5ccefa50031e2f04be06b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1763, 4, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2d0c198538144bff4566e6d94888c1a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a43f2bb9a9a5ccefa50031e2f04be06b
    def get_inputs(self):
        return [
            paddle.uniform([1763, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2d0c198538144bff4566e6d94888c1a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a43f2bb9a9a5ccefa50031e2f04be06b
    def get_inputs(self):
        return [
            paddle.uniform([1763, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_04352ed1167f62cc59c7ae025a0fe812(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 336, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e57ebc6755fe098cb0d39a776554398e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04352ed1167f62cc59c7ae025a0fe812
    def get_inputs(self):
        return [
            paddle.uniform([22, 336, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_a008350b73210386c51d21f8a693b9b5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 4116, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_80b75197bb45c915a8cc9ff953db2f70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a008350b73210386c51d21f8a693b9b5
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4116, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_775cbbbf39499bad646c6bb77979a477(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 240, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dddf9f32a07f90564254f5ce0410d139(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_775cbbbf39499bad646c6bb77979a477
    def get_inputs(self):
        return [
            paddle.uniform([171, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2fc907fe8015667ac84f244a953c12a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d43bcbbd77d19bf34e98d9d221d34d
    def get_inputs(self):
        return [
            paddle.uniform([171, 336, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_2978da7fcfddc97bb6df6fd4381b1388(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 1536, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_287b05b91daf1f42c8a99a04be1e2049(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2978da7fcfddc97bb6df6fd4381b1388
    def get_inputs(self):
        return [
            paddle.uniform([22, 1536, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_b622bb6b2478c6b2dd31678b724ee5e0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[24, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fea0871e20f342f5ec0dab7592e89caf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b622bb6b2478c6b2dd31678b724ee5e0
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.2834267616271973], [1.9475374221801758], [1.9306038618087769], [2.1778082847595215], [2.1797707080841064], [2.006352424621582], [2.2974331378936768], [2.184471607208252], [2.006229877471924], [1.9248852729797363], [2.1701712608337402], [2.1094906330108643], [2.056558609008789], [2.187903881072998], [2.147904396057129], [1.9835880994796753], [2.387515068054199], [1.9674217700958252], [1.884556531906128], [2.200183391571045], [1.9284087419509888], [2.000828266143799], [2.1119980812072754], [2.121765375137329]], dtype='float32').reshape([24, 1]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_aecde0a370bb3a584b9dbe992ae55102(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b622bb6b2478c6b2dd31678b724ee5e0
    def get_inputs(self):
        return [
            paddle.to_tensor([[1.9443585872650146], [1.8843632936477661], [2.3338170051574707], [2.1591877937316895], [2.19769024848938], [2.2046902179718018], [1.9113688468933105], [2.0742058753967285], [2.0201146602630615], [1.9687464237213135], [2.163201332092285], [2.0628907680511475], [2.059060573577881], [2.096897602081299], [2.2119154930114746], [2.073439121246338], [2.087416648864746], [2.0209898948669434], [2.013093948364258], [2.137061357498169], [2.1214048862457275], [2.169085741043091], [1.9357800483703613], [2.0269126892089844]], dtype='float32').reshape([24, 1]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d43a639901560ead81009834774e51cb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 60, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4b67b24aaac2407fac66c0a7de6cae7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d43a639901560ead81009834774e51cb
    def get_inputs(self):
        return [
            paddle.uniform([171, 60, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_42c6c151434b8c0c4e6401de36467872(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 6069, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8e0f97d2ea7cef827a7f7218c191d5b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42c6c151434b8c0c4e6401de36467872
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 6069, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ae6bbb3fbd7aae6b28df6f314e8565a2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1490, 4, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2ff096f7650c641edf36514e13d7421c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae6bbb3fbd7aae6b28df6f314e8565a2
    def get_inputs(self):
        return [
            paddle.uniform([1490, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2ff096f7650c641edf36514e13d7421c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae6bbb3fbd7aae6b28df6f314e8565a2
    def get_inputs(self):
        return [
            paddle.uniform([1490, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_289b98422891117e27909369eceb2492(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 240, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6a5db847d8a72458b732ffdee4ec1b19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_289b98422891117e27909369eceb2492
    def get_inputs(self):
        return [
            paddle.uniform([22, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_cbfc426f2b6a859f8234b70644320e7b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 1536, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_894cb239427b52b0e87e70f6e2880b9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbfc426f2b6a859f8234b70644320e7b
    def get_inputs(self):
        return [
            paddle.uniform([10, 1536, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_e1e53495d1fb1aaae8f0381c276bc9ba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1a4d42c07985e9e52537527842362201(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1e53495d1fb1aaae8f0381c276bc9ba
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.1882402896881104], [2.052551746368408], [1.9094576835632324], [1.9341490268707275]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ffae2ca90f3065dc40327090a3a52786(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1e53495d1fb1aaae8f0381c276bc9ba
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.070204973220825], [2.035569667816162], [2.0479705333709717], [2.227886438369751]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_2e56738512e307f8f65b623881345b88(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 1, 70, 134], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_54427c905ed56f17ab3a0f0b38926034(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2e56738512e307f8f65b623881345b88
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 70, 134], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_3500022561c79371e70f0fc8aa1191ce(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 1, 104, 101], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ccdf808e42beda624d727180d525db63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3500022561c79371e70f0fc8aa1191ce
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 104, 101], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_cf225dca41c935cc5f3b6775060e85cf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2204, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e7f340706089747d59f1bae5a0c85de4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf225dca41c935cc5f3b6775060e85cf
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2204, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_49af5e418baf29762b5dd5a05a688ecf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 36, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5d670658d78074843b6650244f960e3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49af5e418baf29762b5dd5a05a688ecf
    def get_inputs(self):
        return [
            paddle.uniform([22, 36, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_b08107d52a2a7d2409eeb7feaf780639(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 1, 68, 132], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_739d4a983c655c15c10e55e923962bcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b08107d52a2a7d2409eeb7feaf780639
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 68, 132], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_9e9a0b73254099a16c93638581a3ec2a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 1000, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_37837cc675267f6e7ff3c836609e3e14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e9a0b73254099a16c93638581a3ec2a
    def get_inputs(self):
        return [
            paddle.uniform([11, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_9d52aba8b65ab115df7858d8feea5840(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 60, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_282d7fcb19cd372990676d5f1cbc92e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d52aba8b65ab115df7858d8feea5840
    def get_inputs(self):
        return [
            paddle.uniform([145, 60, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_2290f7210979ab10abd6b6a81d84f4a1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 36, 32, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_080927e21ef0af8efc83d6a6cd500fd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2290f7210979ab10abd6b6a81d84f4a1
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 36, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_080927e21ef0af8efc83d6a6cd500fd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2290f7210979ab10abd6b6a81d84f4a1
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 36, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ff1a1db33d8c691374afee64e2bebbda(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[70, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_89afee8bf7d72c4bfaf5178cda9d91bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff1a1db33d8c691374afee64e2bebbda
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[70, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_57aa55d9ccfff45556b5a67dc13ebaea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 672, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fe0e8b92e3123f12503127ee52168b52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57aa55d9ccfff45556b5a67dc13ebaea
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_f521204aa58923b119593741469c40c0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[551, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_302cb47924eb10d6354f4bbe81af9fa5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f521204aa58923b119593741469c40c0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[551, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e57ebc6755fe098cb0d39a776554398e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04352ed1167f62cc59c7ae025a0fe812
    def get_inputs(self):
        return [
            paddle.uniform([22, 336, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_c320c6fe310b31774f91104e6a434386(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[247, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2c10ed9f3422964473f648c54a608f2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c320c6fe310b31774f91104e6a434386
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[247, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_2ec3013fb265409d64f7cb1c1d004d1b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2048, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e6c6cf39d9f2e4eddb1b876f3fbcda53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ec3013fb265409d64f7cb1c1d004d1b
    def get_inputs(self):
        return [
            paddle.uniform([10, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_295fab55ca90bb2a348510081bbeb36f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[950, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b3097e1fc310cc95defd3d0baccd667f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_295fab55ca90bb2a348510081bbeb36f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[950, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c8add2e9f7d40060abe1a65d521e7925(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2076, 4, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_98982425c048cf691272733da283d54c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8add2e9f7d40060abe1a65d521e7925
    def get_inputs(self):
        return [
            paddle.uniform([2076, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_98982425c048cf691272733da283d54c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8add2e9f7d40060abe1a65d521e7925
    def get_inputs(self):
        return [
            paddle.uniform([2076, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_91694dbb809c8a50a67e38b6f1570cbe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8816, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6d91b2233c3677a47b89ce7553558c10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91694dbb809c8a50a67e38b6f1570cbe
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[8816, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_223aa9e47e921e271b504af662c4ea1c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4642, 4, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d03389d454e9cde657677854c386ccd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_223aa9e47e921e271b504af662c4ea1c
    def get_inputs(self):
        return [
            paddle.uniform([4642, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d03389d454e9cde657677854c386ccd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_223aa9e47e921e271b504af662c4ea1c
    def get_inputs(self):
        return [
            paddle.uniform([4642, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_48edc88f3d8ea692816deb12e66c4762(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 96, 1, 40], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ce5ccded3b9c22659200d3e1d7d63eb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48edc88f3d8ea692816deb12e66c4762
    def get_inputs(self):
        return [
            paddle.uniform([10, 96, 1, 40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_865a41e8aa4d1f77ee9743e83976a524(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2434, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c1e240c1d9f81fde384cfa21301c8cf6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_865a41e8aa4d1f77ee9743e83976a524
    def get_inputs(self):
        return [
            paddle.uniform([1, 2434, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f81f19d9db4af65b7b643209c032c205(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2434, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_99bf0a0ab0a1a709d1abbe6cb6439af1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f81f19d9db4af65b7b643209c032c205
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 2434, 1], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_941329043552278ab8952d7fdcca2efc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1047, 4, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_683927d68dea3f236cfd5050dbb17a63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_941329043552278ab8952d7fdcca2efc
    def get_inputs(self):
        return [
            paddle.uniform([1047, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_683927d68dea3f236cfd5050dbb17a63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_941329043552278ab8952d7fdcca2efc
    def get_inputs(self):
        return [
            paddle.uniform([1047, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_817dcb7b748510cfeb6bc757ea5dd175(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 9261, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_330b519a9e8b7685023e0ee2c968a54d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_817dcb7b748510cfeb6bc757ea5dd175
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 9261, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_b1203c433a97de8f83aba7d8365d90db(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 768, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7213586721e537ab1e1f96287af97adc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1203c433a97de8f83aba7d8365d90db
    def get_inputs(self):
        return [
            paddle.uniform([11, 768, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_9a1defbd201d4f3a5d162d33ff285ef5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 1, 64, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c06f419519a8a82b50965852ccfe910a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a1defbd201d4f3a5d162d33ff285ef5
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_bce43fdfbe9bab810a94a19c10b95a1c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 1000, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b2b9240abf214cd5aaa7b44495be2bdd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bce43fdfbe9bab810a94a19c10b95a1c
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_bd2ebe0f9bbea218e59ca1f4df15092e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 1000, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b342ab8b5329ed4021a792ddc9681a86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd2ebe0f9bbea218e59ca1f4df15092e
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_9cec672e99701d39b8c750d7d61fe1c7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 2100, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0ec415f10aa779285da3d89c8d9ffea4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9cec672e99701d39b8c750d7d61fe1c7
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2100, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ea4f962ae20a5c30e3c060469e690712(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1248, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e0eb4722812d0253b25a88f84c0e6e87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea4f962ae20a5c30e3c060469e690712
    def get_inputs(self):
        return [
            paddle.uniform([1, 1248, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_d03b6e2a6f3763f8d8295a1bdaf865e6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 480, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c558c99b795c8efee202f6d63056653b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d03b6e2a6f3763f8d8295a1bdaf865e6
    def get_inputs(self):
        return [
            paddle.uniform([171, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_5d50e7f2ecb9e680b2ba65cd2420aed0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 36, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_63ae1fbd259c68d24dd857053bfb2b58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d50e7f2ecb9e680b2ba65cd2420aed0
    def get_inputs(self):
        return [
            paddle.uniform([145, 36, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_13808056e047eea680f6bb459e239482(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 9, 128, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9152cda3c9e28a4b78bf41d8026e3919(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13808056e047eea680f6bb459e239482
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 9, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9152cda3c9e28a4b78bf41d8026e3919(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13808056e047eea680f6bb459e239482
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 9, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_46f94c4c350eeca74aa5a58678a407bb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2359, 4, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8d08883000b5d74c7a0739d7d6ffa83b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46f94c4c350eeca74aa5a58678a407bb
    def get_inputs(self):
        return [
            paddle.uniform([2359, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8d08883000b5d74c7a0739d7d6ffa83b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46f94c4c350eeca74aa5a58678a407bb
    def get_inputs(self):
        return [
            paddle.uniform([2359, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ae532c232f9abccd30f8500c4666fca6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 96, 32, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e19f002cd07cafe7187bea5943f5096c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae532c232f9abccd30f8500c4666fca6
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 96, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e19f002cd07cafe7187bea5943f5096c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae532c232f9abccd30f8500c4666fca6
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 96, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_42cd958c256e36b56da731dd3254978f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3049, 4, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e2c555f71f737c0b3e823ae7d2bfa5ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cd958c256e36b56da731dd3254978f
    def get_inputs(self):
        return [
            paddle.uniform([3049, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e2c555f71f737c0b3e823ae7d2bfa5ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cd958c256e36b56da731dd3254978f
    def get_inputs(self):
        return [
            paddle.uniform([3049, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c8c63b6106146fc836ba2b1ad3f5276d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3806, 4, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d297e47d2d8e888b6b1b0dba1b680c62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8c63b6106146fc836ba2b1ad3f5276d
    def get_inputs(self):
        return [
            paddle.uniform([3806, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d297e47d2d8e888b6b1b0dba1b680c62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8c63b6106146fc836ba2b1ad3f5276d
    def get_inputs(self):
        return [
            paddle.uniform([3806, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_1fe8ff07c55baedc86dc1e6df129433b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 24, 128, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0d6283988af15451f0c6ded8e4ee456b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fe8ff07c55baedc86dc1e6df129433b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 24, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0d6283988af15451f0c6ded8e4ee456b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fe8ff07c55baedc86dc1e6df129433b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 24, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_7e37271b9d3250d8b993a2b1158f7781(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 156, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_57c499edc16743be9e34f63f40f4fdca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e37271b9d3250d8b993a2b1158f7781
    def get_inputs(self):
        return [
            paddle.uniform([1, 156, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_acb19a3a1770f6511aca2c21c26c97e9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 48, 64, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2745728fc96948fd51c787b51c0a8ded(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_acb19a3a1770f6511aca2c21c26c97e9
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 48, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2745728fc96948fd51c787b51c0a8ded(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_acb19a3a1770f6511aca2c21c26c97e9
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 48, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_9d70698dd69cb54f3eb6d90a4e576b2b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 11109, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_40d8c426b40adaa490717fe34934fb05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d70698dd69cb54f3eb6d90a4e576b2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 11109, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c9c49846614229d1c2568dccd8e10757(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c16979f803ad9cfb857a8b9ca142747a
    def get_inputs(self):
        return [
            paddle.uniform([1, 872, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_aab042d7bd7b9e4da1a86afc5f82fd05(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 480, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7d155cbb3d1a8be1204533aa8911e4c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aab042d7bd7b9e4da1a86afc5f82fd05
    def get_inputs(self):
        return [
            paddle.uniform([22, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_48836454ac352a3205a727c248a6a53c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 480, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7aa635c9c3755a299118c87b5703032b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48836454ac352a3205a727c248a6a53c
    def get_inputs(self):
        return [
            paddle.uniform([145, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_f568f64f3184052a9d2139dec2583a2b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 192, 1, 25], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_17c7c9a678b7ee33e986a8fcc5acebea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f568f64f3184052a9d2139dec2583a2b
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 1, 25], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_435f3702be15824d884eb8be5535aaee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 36, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_683e5eef5db755d2c1f84667915e057c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_435f3702be15824d884eb8be5535aaee
    def get_inputs(self):
        return [
            paddle.uniform([171, 36, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_8c2cc6a75ebc5be0e84e2c29e3df2ed0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 120, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a6c40624af36f08d832bac1169556289(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c2cc6a75ebc5be0e84e2c29e3df2ed0
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_f2f5bd0124a148b0eccf21c56216cfe4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_adf3c8a8167235613d69b2c778137844(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f2f5bd0124a148b0eccf21c56216cfe4
    def get_inputs(self):
        return [
            paddle.to_tensor([[1.9969403743743896], [2.232224225997925], [2.1616365909576416], [2.382333755493164], [1.9233232736587524], [2.1826717853546143], [2.188194751739502], [2.2212114334106445], [1.9976115226745605], [1.9273639917373657], [2.208949089050293], [1.9189504384994507], [2.073084831237793], [1.8286184072494507], [2.136892080307007], [1.8656898736953735], [1.8957819938659668], [1.969542384147644], [2.1494929790496826], [2.1369214057922363]], dtype='float32').reshape([20, 1]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fd6d9d0572cb1d1771584acf87226f41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f2f5bd0124a148b0eccf21c56216cfe4
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.3048977851867676], [2.116800308227539], [2.23038649559021], [2.011624813079834], [2.2954282760620117], [1.9638127088546753], [2.1289992332458496], [2.2716681957244873], [1.912353754043579], [2.214439868927002], [1.949325680732727], [2.3215630054473877], [2.042908191680908], [2.256613254547119], [2.226163625717163], [2.172593593597412], [2.1633942127227783], [2.1945648193359375], [2.174997329711914], [2.17669939994812]], dtype='float32').reshape([20, 1]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2c10ed9f3422964473f648c54a608f2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c320c6fe310b31774f91104e6a434386
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[247, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fe0e8b92e3123f12503127ee52168b52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57aa55d9ccfff45556b5a67dc13ebaea
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d314a816fb6797f3c44fe3cf8061faef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13e304be87067a63d9b39c3b79a741f2
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3800, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_64814c4c85176629deda855e353b897c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8732, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_448ad977021dc4392972ee5e843997fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64814c4c85176629deda855e353b897c
    def get_inputs(self):
        return [
            paddle.uniform([1, 8732, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_741aef161a1aaf31d14690312ba1b77d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8732, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dbe1b547080af5bcf46ddf1024d4876c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_741aef161a1aaf31d14690312ba1b77d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 8732, 1], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b3097e1fc310cc95defd3d0baccd667f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_295fab55ca90bb2a348510081bbeb36f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[950, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_235174c26db5683f9090a8ed78acfba8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2054, 4, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1511b76b3382857793e07d691b6e22a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_235174c26db5683f9090a8ed78acfba8
    def get_inputs(self):
        return [
            paddle.uniform([2054, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1511b76b3382857793e07d691b6e22a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_235174c26db5683f9090a8ed78acfba8
    def get_inputs(self):
        return [
            paddle.uniform([2054, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_14cd11329a64593629ab60d69c11bb84(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 1000, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c88ad97074c7843d94035d5c0abc0954(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_14cd11329a64593629ab60d69c11bb84
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_89afee8bf7d72c4bfaf5178cda9d91bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff1a1db33d8c691374afee64e2bebbda
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[70, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_7e55a280321dc265c32174337d5b06f6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 3024, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9c3d832bcf55aa8e971a91c12aa3975c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e55a280321dc265c32174337d5b06f6
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 3024, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_bd19e090ade2cbaa12d1f559785171ba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 1280, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_98323dc5050361be9a643841bd1ae801(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd19e090ade2cbaa12d1f559785171ba
    def get_inputs(self):
        return [
            paddle.uniform([11, 1280, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_a2660044561cf057b1c65f8321a40b8b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4218, 4, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_89d69d0b708ca7b6c96883c547a22aba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2660044561cf057b1c65f8321a40b8b
    def get_inputs(self):
        return [
            paddle.uniform([4218, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_89d69d0b708ca7b6c96883c547a22aba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2660044561cf057b1c65f8321a40b8b
    def get_inputs(self):
        return [
            paddle.uniform([4218, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_761958b6d904545f5fc281328b6e0544(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 624, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f837e3a09cb9e065a9a25e7bd71116a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_761958b6d904545f5fc281328b6e0544
    def get_inputs(self):
        return [
            paddle.uniform([1, 624, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_a810759bde6368e889ff3a57ab926a2f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 1000, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3667b1d878b832a51f9ed93ca11ff62d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a810759bde6368e889ff3a57ab926a2f
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_e6f1ea95e355d4e33c4507f6c3f93e51(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 1000, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bf67477a65e6981c568af9b13c0aa465(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6f1ea95e355d4e33c4507f6c3f93e51
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()