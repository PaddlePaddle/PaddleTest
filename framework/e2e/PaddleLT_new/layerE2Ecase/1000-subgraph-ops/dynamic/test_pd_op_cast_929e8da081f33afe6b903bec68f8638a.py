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



class PrimitiveOp_05c68fcefee09b367cab4d626ad01c5e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_85e11fc25207591dacc37fef83270607(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05c68fcefee09b367cab4d626ad01c5e
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_c785908ecb825189e48322e59b25894b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_43aa039cfcd3b9ee2fe2060a9ed9473c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 3024, 4], dtype='int32'),
        ]


class PrimitiveOp_3e0d820df191fb74bb4c161cb34c6478(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_660339df004dadd56ae4afdc9455efb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e0d820df191fb74bb4c161cb34c6478
    def get_inputs(self):
        return [
            paddle.to_tensor(6096, dtype='int32').reshape([]),
        ]


class PrimitiveOp_a7b3da5ca116a3c8cd2602d1697852c4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8e18738ec19410f32fcaf1251c7b5886(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7b3da5ca116a3c8cd2602d1697852c4
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_bf7ace93bc8e08f111c51ae603f671dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 3024, 68], dtype='int32'),
        ]


class PrimitiveOp_0bf33482bd49195c5fe8c0182d505424(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d52ba42d2aac676bed0d5115a7dc55c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bf33482bd49195c5fe8c0182d505424
    def get_inputs(self):
        return [
            paddle.uniform([1524, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0038e1b2949050eeb1c879e5d3c72857(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e1cd30755c6ffb1c5e2059b9cb2c726a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0038e1b2949050eeb1c879e5d3c72857
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1524, 4], dtype='int64'),
        ]


class TestPrimitiveOp_7856772ebd11dbd2ef8d56ee395b28d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05c68fcefee09b367cab4d626ad01c5e
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_856095ce6442486f7f5da239753856af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 4725, 4], dtype='int32'),
        ]


class TestPrimitiveOp_63b1dd9d7591ceb04845630d612c81a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e0d820df191fb74bb4c161cb34c6478
    def get_inputs(self):
        return [
            paddle.to_tensor(9360, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_fc302595cbcae47566bacaace7e2b68e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7b3da5ca116a3c8cd2602d1697852c4
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_49289a34c5c4b4fc0326779f978e6589(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 4725, 68], dtype='int32'),
        ]


class TestPrimitiveOp_1606339b0fc84b7148f56c6ee518b625(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bf33482bd49195c5fe8c0182d505424
    def get_inputs(self):
        return [
            paddle.uniform([2340, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a08c6a3167106cae6fa4f4e358ac3d44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0038e1b2949050eeb1c879e5d3c72857
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2340, 4], dtype='int64'),
        ]


class PrimitiveOp_67329cbfbfe6e2e657262692edbd53a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e68e21bb2938714cb20dad4be8e4a2f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67329cbfbfe6e2e657262692edbd53a3
    def get_inputs(self):
        return [
            paddle.to_tensor([100.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_78fa2a50088b0f2c44f61257ed686456(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3b8d9a8166c0f8fcf64220b6a289d791(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_225cd86d0a147fc4afa8a03fd9394229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(512, dtype='int32').reshape([]),
        ]


class PrimitiveOp_445147eb008d0af23b194c80d911dcfa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_387acbe7cdf3a7be7832244a006f018f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_445147eb008d0af23b194c80d911dcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_614fd400b8b2ac377a78a522c72b3e14(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_826a90626ff39f464ae13a6660316480(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_614fd400b8b2ac377a78a522c72b3e14
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_b8cd3eac4b0759bcf033ac4911afa05c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_acbcbcebfcb12fc277cf179be8629ebf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8cd3eac4b0759bcf033ac4911afa05c
    def get_inputs(self):
        return [
            paddle.to_tensor([False, False, False, False, False, False], dtype='bool').reshape([6]),
        ]


class TestPrimitiveOp_721d998005c0d16ddd773c0ed6aaab55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8cd3eac4b0759bcf033ac4911afa05c
    def get_inputs(self):
        return [
            paddle.to_tensor([True, True, False, True, False, False], dtype='bool').reshape([6]),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_7bc8eeffbbc886c484fdfc552b626716(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(256, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_07a6a388692472a606723abbe969916d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_445147eb008d0af23b194c80d911dcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f16aeb04493caaa635761e0dc5bd152b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_614fd400b8b2ac377a78a522c72b3e14
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 256, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_3b8d9a8166c0f8fcf64220b6a289d791(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_3d3e8259dcc44645938377990f0f082e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_445147eb008d0af23b194c80d911dcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_be1b842380ad18e922bf36eaaccde450(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_614fd400b8b2ac377a78a522c72b3e14
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 64, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_3d3e8259dcc44645938377990f0f082e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_445147eb008d0af23b194c80d911dcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_be1b842380ad18e922bf36eaaccde450(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_614fd400b8b2ac377a78a522c72b3e14
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 64, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_9ebf0cf29337c314ccf94b913a219ae5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(192, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_74516a96b6cebc612396f6cf5bd50ce4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_445147eb008d0af23b194c80d911dcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d6d54d5babe998459821b5febccfe4c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_614fd400b8b2ac377a78a522c72b3e14
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 192, 1, 1], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_4888e346918d3535335f3532e1b10a11(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_77ac76b65559835b337a4fef0166f7bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4888e346918d3535335f3532e1b10a11
    def get_inputs(self):
        return [
            paddle.to_tensor([0.414253294467926, 0.40653812885284424, 0.39356136322021484, 0.39688950777053833], dtype='float32').reshape([4]),
        ]


class TestPrimitiveOp_8bbdca277bfb57d348a1afe47e48507a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f0d1762971dbbc8d54f7af458b2fd659(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_1000fbcf3b664467ab543f04708f3ef6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[152], dtype='int64'),
        ]


class TestPrimitiveOp_8706efe4c33bd88122dc407e953c30ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[100], dtype='int64'),
        ]


class PrimitiveOp_74168d360d4449d2f23e4a840a6867b7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_23b8916f335b2a43798f891ef92971fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([100, 152, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7da86ca5a97d61079c717d9105b39020(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([100, 152, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7903ec8a46a704a418ca1c3dc0d9b99f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[76], dtype='int64'),
        ]


class TestPrimitiveOp_bae2c2d93921e4542b5cfc2fa11d319c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[50], dtype='int64'),
        ]


class TestPrimitiveOp_4a46892ec94ea0a60f686e223c17a5fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([50, 76, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c557223f8f587567667bc9d819affc53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([50, 76, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1f096a1528e6fea9bca8f3c814dd243a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[38], dtype='int64'),
        ]


class TestPrimitiveOp_1db89c6758a94975e33fd7969904c972(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], dtype='int64').reshape([25]),
        ]


class TestPrimitiveOp_08e84da629fec5f874b27232d1c85135(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([25, 38, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a2685efa453ec59f7b24da69d39bbebb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([25, 38, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_07eee576f977a434a9e5afee1bee8f4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], dtype='int64').reshape([19]),
        ]


class TestPrimitiveOp_1133e468407977ab877c4d48248ddd70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype='int64').reshape([13]),
        ]


class TestPrimitiveOp_034e98f669fc93fad8c9089de739414e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([13, 19, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b383e3086fc57f8ea81db703bea37040(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([13, 19, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_191e71de83910e9f8176910b647b2a0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='int64').reshape([10]),
        ]


class TestPrimitiveOp_cb8d87c6b8c2bb5eadd5a99dc79a5a37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6], dtype='int64').reshape([7]),
        ]


class TestPrimitiveOp_49e33338ce7b33db30d506f327d2d07e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([7, 10, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_87d8eff786878f8e5f262234fc918e4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([7, 10, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5ea569821d90632c221d87d698d47aad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9ee71a801e379743fcdf0fe71f22ccf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67329cbfbfe6e2e657262692edbd53a3
    def get_inputs(self):
        return [
            paddle.to_tensor([300.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9ee71a801e379743fcdf0fe71f22ccf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67329cbfbfe6e2e657262692edbd53a3
    def get_inputs(self):
        return [
            paddle.to_tensor([300.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7911037999e5f9d9846dbf02fe637b0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05c68fcefee09b367cab4d626ad01c5e
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_50d44f1f18014a126afd22af04d946a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'),
        ]


class TestPrimitiveOp_402c357c9f88fc1a86d4d677d92405af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e0d820df191fb74bb4c161cb34c6478
    def get_inputs(self):
        return [
            paddle.to_tensor(8188, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_145a6d15ba2fcc9ee0ff11e2c73bc351(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7b3da5ca116a3c8cd2602d1697852c4
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_191ec2501a8ec43f40e4e2c294ee41cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 4116, 68], dtype='int32'),
        ]


class TestPrimitiveOp_736a12ba7ce3557973d6df5b3f4d67b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bf33482bd49195c5fe8c0182d505424
    def get_inputs(self):
        return [
            paddle.uniform([2047, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2b727b9785fd495e910660b995b054bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0038e1b2949050eeb1c879e5d3c72857
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2047, 4], dtype='int64'),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c6309e4495c24ad71d476ccb6410e242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_573157f0a440dc84e884fedb52019ad0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_445147eb008d0af23b194c80d911dcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8d1c53dc389badbafed1347ab6297f82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_614fd400b8b2ac377a78a522c72b3e14
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 128, 1, 1], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_f57cf0bd37c3c802a6a37943aa1ae7ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_df0c1fe2e0d820318c8dc05df409edee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f57cf0bd37c3c802a6a37943aa1ae7ef
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[256], dtype='int32'),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_7bc8eeffbbc886c484fdfc552b626716(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(256, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_d3287cd44094a52d74ea6ad9731ef256(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_445147eb008d0af23b194c80d911dcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 256, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f16aeb04493caaa635761e0dc5bd152b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_614fd400b8b2ac377a78a522c72b3e14
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 256, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_032f739c474d96cc12983ccd8e618279(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[80], dtype='int64'),
        ]


class TestPrimitiveOp_1d48b31e1c11cbeee6ccd4966d73bc56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[40], dtype='int64'),
        ]


class TestPrimitiveOp_4ada8ff78142a496c452c96a035ae227(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype='int64').reshape([20]),
        ]


class PrimitiveOp_2cceba588efb2cce6310a59179db9450(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_52dbde8202a6a318f79f876a62655378(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cceba588efb2cce6310a59179db9450
    def get_inputs(self):
        return [
            paddle.uniform([8400, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_52dbde8202a6a318f79f876a62655378(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cceba588efb2cce6310a59179db9450
    def get_inputs(self):
        return [
            paddle.uniform([8400, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_396afbf9c2d0a9a3d22b8d829ddc96f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4888e346918d3535335f3532e1b10a11
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.45301729440689087, 0.20257097482681274, -0.1712852418422699, -0.07046157121658325, -0.3312643766403198, -0.1228134036064148, 0.10590583086013794, -0.14092087745666504, -0.43386006355285645, 0.28023451566696167, 0.22156333923339844, -0.4111160337924957, -0.07768765091896057, 0.4597037434577942, 0.24061691761016846, 0.12023502588272095, 0.42479991912841797, -0.36824023723602295, 0.053500473499298096, 0.22389835119247437], dtype='float32').reshape([20]),
        ]


class TestPrimitiveOp_7268aa07933dab324897a95f7aabcece(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([20]),
        ]


class TestPrimitiveOp_c11b327b5370b1d5b7dc5cee55275784(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([20]),
        ]


class TestPrimitiveOp_cdb44a5865fd80087f1e482f2e14436a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05c68fcefee09b367cab4d626ad01c5e
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_e95ecc3f83799dc013d3587b376e7248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'),
        ]


class TestPrimitiveOp_ebe5d4336f8f6604f5e35a43f4b68657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e0d820df191fb74bb4c161cb34c6478
    def get_inputs(self):
        return [
            paddle.to_tensor(7252, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_3091f2efa85af2ed7062532b7cf4f8c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7b3da5ca116a3c8cd2602d1697852c4
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_5068a510b188fdcbcd46239746dc2859(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 3549, 76], dtype='int32'),
        ]


class TestPrimitiveOp_0fe8bbc44dd9308b75fa91ca5fef726b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bf33482bd49195c5fe8c0182d505424
    def get_inputs(self):
        return [
            paddle.uniform([1813, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7658c0de492f7521bf91b03400e8b080(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0038e1b2949050eeb1c879e5d3c72857
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1813, 4], dtype='int64'),
        ]


class TestPrimitiveOp_1bda6188f624214d02533242296232b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4888e346918d3535335f3532e1b10a11
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.2985445261001587, 0.43575096130371094, -0.1309654712677002, -0.3248171806335449, 0.11053144931793213, -0.34735989570617676, 0.33360379934310913, -0.35473915934562683, -0.47531527280807495, -0.4501453936100006, 0.4342665672302246, -0.17041495442390442, -0.05690556764602661, 0.499264121055603, -0.1997799277305603, 0.24060136079788208], dtype='float32').reshape([16]),
        ]


class TestPrimitiveOp_8dac2ca19749ea58f5e799dc0fbb6fcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([16]),
        ]


class TestPrimitiveOp_5cbd0617d5aa9e78ba192b70b5ad728c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([16]),
        ]


class PrimitiveOp_591250b83c1a2579ee9a4464305bddca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8f10be2394c7224e793ec9ff1067ddf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_591250b83c1a2579ee9a4464305bddca
    def get_inputs(self):
        return [
            paddle.to_tensor([0.6515615582466125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_91a6357a931011d7245c3f8a855d6563(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], dtype='int64').reshape([14]),
        ]


class TestPrimitiveOp_8cd4e4e705de2e3f9d2bef61f0455812(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([14, 14, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0c6744c26b2d2ffe5b757a70b001fc37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([14, 14, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_13a66480c01d673fe0216b7fbfad03b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], dtype='int64').reshape([28]),
        ]


class TestPrimitiveOp_3cd62b73948e360e84b32920993a5516(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([28, 28, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0a7f7d17403a85ef8299fe32d5f43c52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([28, 28, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dcbe00243fe4985b7af6bbae22ce1831(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[56], dtype='int64'),
        ]


class TestPrimitiveOp_46008ee63452d35e61533d289ae02d61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([56, 56, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc180615f3595e8367d8b20d490edf45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([56, 56, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f1a150d5ccc6ab419104f692abc582f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05c68fcefee09b367cab4d626ad01c5e
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_edcbf7afc063bc3afed4a0971a498e6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 6069, 4], dtype='int32'),
        ]


class TestPrimitiveOp_b74ceee0153fb2624dbcf1c106b49607(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e0d820df191fb74bb4c161cb34c6478
    def get_inputs(self):
        return [
            paddle.to_tensor(12244, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_30ec09524891046ae2be7796c7789d0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7b3da5ca116a3c8cd2602d1697852c4
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_615158929197d4c5782e43de7b0cef86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 6069, 68], dtype='int32'),
        ]


class TestPrimitiveOp_bb2f4db3278ac3e2564d438d3f0e3eb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bf33482bd49195c5fe8c0182d505424
    def get_inputs(self):
        return [
            paddle.uniform([3061, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f6a1f93e87b681388617288bcbbfec57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0038e1b2949050eeb1c879e5d3c72857
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[3061, 4], dtype='int64'),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c6309e4495c24ad71d476ccb6410e242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_ecb7014ad6137966dbfb400b7e135b97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_445147eb008d0af23b194c80d911dcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8d1c53dc389badbafed1347ab6297f82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_614fd400b8b2ac377a78a522c72b3e14
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 128, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_7911037999e5f9d9846dbf02fe637b0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05c68fcefee09b367cab4d626ad01c5e
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_50d44f1f18014a126afd22af04d946a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'),
        ]


class TestPrimitiveOp_9a85b4cf3cc67fe5e7f691e3e29d06ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e0d820df191fb74bb4c161cb34c6478
    def get_inputs(self):
        return [
            paddle.to_tensor(8248, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_145a6d15ba2fcc9ee0ff11e2c73bc351(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7b3da5ca116a3c8cd2602d1697852c4
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_191ec2501a8ec43f40e4e2c294ee41cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 4116, 68], dtype='int32'),
        ]


class TestPrimitiveOp_81d51d265e8333c01ff3ff588825a837(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bf33482bd49195c5fe8c0182d505424
    def get_inputs(self):
        return [
            paddle.uniform([2062, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ac6f781bdeaea640a859af57acdb81f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0038e1b2949050eeb1c879e5d3c72857
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2062, 4], dtype='int64'),
        ]


class TestPrimitiveOp_81e92b6774203d74c0434b566b8f498c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([7], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7c4a8f870ab1df6757710a20d9cd82d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2d24d653e54fa60a1a4d4bd0b61bf500(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[32], dtype='int64'),
        ]


class TestPrimitiveOp_dae33979e5cf630b6f0885b047e312eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([32, 32, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e2414645af2492db08f4826e2f92ea39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[64], dtype='int64'),
        ]


class TestPrimitiveOp_ccef7b3f2df1118c22699b674ee11b38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([64, 64, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5e1f093113564222cca350000d8a8ccf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[128], dtype='int64'),
        ]


class TestPrimitiveOp_c68179dd09c19a548ebf9cffce3f81c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([128, 128, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3ea3655edb7d7c87ce9dfa2b8a266ab4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[96], dtype='int64'),
        ]


class TestPrimitiveOp_3be7cf4949fb125a5e4279def078c126(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[48], dtype='int64'),
        ]


class TestPrimitiveOp_b2d7fc7cbb8d914733373119db97ba25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], dtype='int64').reshape([24]),
        ]


class TestPrimitiveOp_ddeb099bb1ebc3b4752d35281eb03aee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cceba588efb2cce6310a59179db9450
    def get_inputs(self):
        return [
            paddle.uniform([12096, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ddeb099bb1ebc3b4752d35281eb03aee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cceba588efb2cce6310a59179db9450
    def get_inputs(self):
        return [
            paddle.uniform([12096, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_225cd86d0a147fc4afa8a03fd9394229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(512, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_840494ed82c73b2d70b4c441b50bc174(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_445147eb008d0af23b194c80d911dcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_826a90626ff39f464ae13a6660316480(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_614fd400b8b2ac377a78a522c72b3e14
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_2d24d653e54fa60a1a4d4bd0b61bf500(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[32], dtype='int64'),
        ]


class TestPrimitiveOp_dae33979e5cf630b6f0885b047e312eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([32, 32, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e2414645af2492db08f4826e2f92ea39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[64], dtype='int64'),
        ]


class TestPrimitiveOp_ccef7b3f2df1118c22699b674ee11b38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([64, 64, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5e1f093113564222cca350000d8a8ccf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[128], dtype='int64'),
        ]


class TestPrimitiveOp_c68179dd09c19a548ebf9cffce3f81c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([128, 128, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_225cd86d0a147fc4afa8a03fd9394229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(512, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_387acbe7cdf3a7be7832244a006f018f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_445147eb008d0af23b194c80d911dcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_826a90626ff39f464ae13a6660316480(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_614fd400b8b2ac377a78a522c72b3e14
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_225cd86d0a147fc4afa8a03fd9394229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(512, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_387acbe7cdf3a7be7832244a006f018f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_445147eb008d0af23b194c80d911dcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_826a90626ff39f464ae13a6660316480(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_614fd400b8b2ac377a78a522c72b3e14
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_225cd86d0a147fc4afa8a03fd9394229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(512, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_387acbe7cdf3a7be7832244a006f018f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_445147eb008d0af23b194c80d911dcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_826a90626ff39f464ae13a6660316480(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_614fd400b8b2ac377a78a522c72b3e14
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c593e8c6b6abeae488761917b767345b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(2048, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_9e05ded3c8c8782e6b188e9d88277ab3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_445147eb008d0af23b194c80d911dcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c358fed45d6cc1da90aa0e60556341e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_614fd400b8b2ac377a78a522c72b3e14
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2048, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_5e91f7a7c1d960c00de92b55e5cf5704(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[68], dtype='int64'),
        ]


class TestPrimitiveOp_416df785fc03c72d91fdb330fe8b9b7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[34], dtype='int64'),
        ]


class TestPrimitiveOp_a46bb26a15c7f9a911730ba19b3f294d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], dtype='int64').reshape([17]),
        ]


class TestPrimitiveOp_27ec74367d3ff2ac03f4bb9ac6be2811(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cceba588efb2cce6310a59179db9450
    def get_inputs(self):
        return [
            paddle.uniform([6069, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_27ec74367d3ff2ac03f4bb9ac6be2811(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cceba588efb2cce6310a59179db9450
    def get_inputs(self):
        return [
            paddle.uniform([6069, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_225cd86d0a147fc4afa8a03fd9394229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(512, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_840494ed82c73b2d70b4c441b50bc174(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_445147eb008d0af23b194c80d911dcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_826a90626ff39f464ae13a6660316480(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_614fd400b8b2ac377a78a522c72b3e14
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_225cd86d0a147fc4afa8a03fd9394229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(512, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_840494ed82c73b2d70b4c441b50bc174(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_445147eb008d0af23b194c80d911dcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_826a90626ff39f464ae13a6660316480(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_614fd400b8b2ac377a78a522c72b3e14
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_225cd86d0a147fc4afa8a03fd9394229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(512, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_840494ed82c73b2d70b4c441b50bc174(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_445147eb008d0af23b194c80d911dcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_826a90626ff39f464ae13a6660316480(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_614fd400b8b2ac377a78a522c72b3e14
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c593e8c6b6abeae488761917b767345b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(2048, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_855195c755b31d92c2e97a9ae6132245(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_445147eb008d0af23b194c80d911dcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c358fed45d6cc1da90aa0e60556341e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_614fd400b8b2ac377a78a522c72b3e14
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2048, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_81e92b6774203d74c0434b566b8f498c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([7], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1011944269c230f9277e2cadc603d61e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05c68fcefee09b367cab4d626ad01c5e
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_5e7314a2ed3de3414864abba8c7b62ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 11109, 4], dtype='int32'),
        ]


class TestPrimitiveOp_22396cf69616eda1731efda8cd360df1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e0d820df191fb74bb4c161cb34c6478
    def get_inputs(self):
        return [
            paddle.to_tensor(22104, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_8e0d3b4c356e967e97e82447e93ace3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7b3da5ca116a3c8cd2602d1697852c4
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_9a59d3e8eeeb5eb20ed5fd531e66ab7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 11109, 68], dtype='int32'),
        ]


class TestPrimitiveOp_206a11242b161e99adddef538743fa68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bf33482bd49195c5fe8c0182d505424
    def get_inputs(self):
        return [
            paddle.uniform([5526, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a9a880389a57160a2ab0400f365c4fe0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0038e1b2949050eeb1c879e5d3c72857
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[5526, 4], dtype='int64'),
        ]


class TestPrimitiveOp_ca8f1d8ab9a000f8d1de6835205a88ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05c68fcefee09b367cab4d626ad01c5e
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_1e9f43532d7ad7c987932de1a6ddfb78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 2100, 4], dtype='int32'),
        ]


class TestPrimitiveOp_d2ab2a9a04b4978596795287eae3e8f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e0d820df191fb74bb4c161cb34c6478
    def get_inputs(self):
        return [
            paddle.to_tensor(4284, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_1dc3a633558b3f9fca23d07a0f403dd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7b3da5ca116a3c8cd2602d1697852c4
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_ff82b13293c40c434719989f88d6fb8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 2100, 68], dtype='int32'),
        ]


class TestPrimitiveOp_120cdd581d2d7c7e606cd7265d5b3c9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bf33482bd49195c5fe8c0182d505424
    def get_inputs(self):
        return [
            paddle.uniform([1071, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_63efcfae5db283b2afc5705bf8e1a629(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0038e1b2949050eeb1c879e5d3c72857
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1071, 4], dtype='int64'),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c6309e4495c24ad71d476ccb6410e242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_ecb7014ad6137966dbfb400b7e135b97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_445147eb008d0af23b194c80d911dcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8d1c53dc389badbafed1347ab6297f82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_614fd400b8b2ac377a78a522c72b3e14
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 128, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_225cd86d0a147fc4afa8a03fd9394229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(512, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_ab462465e2523ab56da2ef4b8977da78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_445147eb008d0af23b194c80d911dcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 97, 97], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_826a90626ff39f464ae13a6660316480(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_614fd400b8b2ac377a78a522c72b3e14
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_2d24d653e54fa60a1a4d4bd0b61bf500(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[32], dtype='int64'),
        ]


class TestPrimitiveOp_dae33979e5cf630b6f0885b047e312eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([32, 32, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e2414645af2492db08f4826e2f92ea39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[64], dtype='int64'),
        ]


class TestPrimitiveOp_ccef7b3f2df1118c22699b674ee11b38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([64, 64, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5e1f093113564222cca350000d8a8ccf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[128], dtype='int64'),
        ]


class TestPrimitiveOp_c68179dd09c19a548ebf9cffce3f81c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([128, 128, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cdb44a5865fd80087f1e482f2e14436a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05c68fcefee09b367cab4d626ad01c5e
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_e95ecc3f83799dc013d3587b376e7248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'),
        ]


class TestPrimitiveOp_adaa45f61092b7a3246fd7c9c0209379(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e0d820df191fb74bb4c161cb34c6478
    def get_inputs(self):
        return [
            paddle.to_tensor(7040, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_3091f2efa85af2ed7062532b7cf4f8c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7b3da5ca116a3c8cd2602d1697852c4
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_bb6a95975b025c6e47b028f9646fd4ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 3549, 68], dtype='int32'),
        ]


class TestPrimitiveOp_f6f8c1f3e15024d7beb99334265a8f9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bf33482bd49195c5fe8c0182d505424
    def get_inputs(self):
        return [
            paddle.uniform([1760, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0957f4fc29c9bd26d373db404d50ad45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0038e1b2949050eeb1c879e5d3c72857
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1760, 4], dtype='int64'),
        ]


class TestPrimitiveOp_df0c1fe2e0d820318c8dc05df409edee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f57cf0bd37c3c802a6a37943aa1ae7ef
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[256], dtype='int32'),
        ]


class TestPrimitiveOp_2d24d653e54fa60a1a4d4bd0b61bf500(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[32], dtype='int64'),
        ]


class TestPrimitiveOp_dae33979e5cf630b6f0885b047e312eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([32, 32, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e2414645af2492db08f4826e2f92ea39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[64], dtype='int64'),
        ]


class TestPrimitiveOp_ccef7b3f2df1118c22699b674ee11b38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([64, 64, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5e1f093113564222cca350000d8a8ccf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[128], dtype='int64'),
        ]


class TestPrimitiveOp_c68179dd09c19a548ebf9cffce3f81c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([128, 128, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_effa6c65e6c8ea2a5ecfca7bec943bf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4888e346918d3535335f3532e1b10a11
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.2453816533088684, -0.3311845660209656, 0.3198009133338928, -0.2272983193397522, -0.4603445529937744, 0.13189810514450073, -0.42916133999824524, 0.338689386844635, 0.4118430018424988, 0.4310181140899658, 0.16426771879196167, 0.19444793462753296, 0.3859570622444153, 0.2425786256790161, -0.40610766410827637, 0.24963635206222534, -0.09066009521484375, 0.08062887191772461, 0.062007904052734375, -0.07036173343658447, 0.013734638690948486, 0.11161541938781738, 0.47627896070480347, -0.2894998788833618], dtype='float32').reshape([24]),
        ]


class TestPrimitiveOp_f1bab6b0b640713b4a3bcc00ab17b2f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([24]),
        ]


class TestPrimitiveOp_f287a511eabbc1558de72998e1503fd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([24]),
        ]


class TestPrimitiveOp_5e1f093113564222cca350000d8a8ccf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[128], dtype='int64'),
        ]


class TestPrimitiveOp_c68179dd09c19a548ebf9cffce3f81c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([128, 128, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e2414645af2492db08f4826e2f92ea39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[64], dtype='int64'),
        ]


class TestPrimitiveOp_ccef7b3f2df1118c22699b674ee11b38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([64, 64, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2d24d653e54fa60a1a4d4bd0b61bf500(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[32], dtype='int64'),
        ]


class TestPrimitiveOp_dae33979e5cf630b6f0885b047e312eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([32, 32, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_56d5378017754b901a36c34d3a89ed40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype='int64').reshape([16]),
        ]


class TestPrimitiveOp_1d5a3f1c455bb4cd4d417f37ab91aa57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([16, 16, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6582ede0a83c426599075291239aa7b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype='int64').reshape([8]),
        ]


class TestPrimitiveOp_fc2912366b4f70227bf4e9cf4f6a42a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([8, 8, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_81e92b6774203d74c0434b566b8f498c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([7], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e68e21bb2938714cb20dad4be8e4a2f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67329cbfbfe6e2e657262692edbd53a3
    def get_inputs(self):
        return [
            paddle.to_tensor([100.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ec80dcac1f1cdf553ba46399dcc25fe0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([8], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5ea569821d90632c221d87d698d47aad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c9823ca4ce4f8796c0132f4760b2227d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05c68fcefee09b367cab4d626ad01c5e
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_7a12ead5b92989816b36beda359b795b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 8400, 4], dtype='int32'),
        ]


class TestPrimitiveOp_28d75756d3499500f27f1755ecba2f9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e0d820df191fb74bb4c161cb34c6478
    def get_inputs(self):
        return [
            paddle.to_tensor(16816, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_17f7717e92a9479f0ca6239434e22ca6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7b3da5ca116a3c8cd2602d1697852c4
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_d1d447fbd56e01ce602ba05816b11765(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 8400, 68], dtype='int32'),
        ]


class TestPrimitiveOp_8e96a8086ecde77908b22d20945e9004(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bf33482bd49195c5fe8c0182d505424
    def get_inputs(self):
        return [
            paddle.uniform([4204, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_158c8775547085921002d051decdb1b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0038e1b2949050eeb1c879e5d3c72857
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[4204, 4], dtype='int64'),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_7bc8eeffbbc886c484fdfc552b626716(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(256, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_f6dd4479074d9f15874339887655255d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_445147eb008d0af23b194c80d911dcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f16aeb04493caaa635761e0dc5bd152b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_614fd400b8b2ac377a78a522c72b3e14
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 256, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_911c431d1970ab28ae36db82ef2a362a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3807d50310abc35cda248cd48725080e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[72], dtype='int64'),
        ]


class TestPrimitiveOp_777ca9997ca07e198aea31ff87514515(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[36], dtype='int64'),
        ]


class TestPrimitiveOp_a43073ccfa34f758c88475da88a7881a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], dtype='int64').reshape([18]),
        ]


class TestPrimitiveOp_a0580e7a8badaf83179c49e70a3cbad1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cceba588efb2cce6310a59179db9450
    def get_inputs(self):
        return [
            paddle.uniform([6804, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a0580e7a8badaf83179c49e70a3cbad1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cceba588efb2cce6310a59179db9450
    def get_inputs(self):
        return [
            paddle.uniform([6804, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e2414645af2492db08f4826e2f92ea39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[64], dtype='int64'),
        ]


class TestPrimitiveOp_2d24d653e54fa60a1a4d4bd0b61bf500(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[32], dtype='int64'),
        ]


class TestPrimitiveOp_56d5378017754b901a36c34d3a89ed40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype='int64').reshape([16]),
        ]


class TestPrimitiveOp_7490b78fb4467cb0d7adf93cfe706409(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cceba588efb2cce6310a59179db9450
    def get_inputs(self):
        return [
            paddle.uniform([5376, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7490b78fb4467cb0d7adf93cfe706409(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cceba588efb2cce6310a59179db9450
    def get_inputs(self):
        return [
            paddle.uniform([5376, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9883fe5e546b0fb651eebb8f6964b485(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5c7032b06f00176674606527880e09a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05c68fcefee09b367cab4d626ad01c5e
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_d25fbf8440cc04ee09cef438c537c941(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 9261, 4], dtype='int32'),
        ]


class TestPrimitiveOp_bb8a938259f3804c17f49f5da3ea262f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e0d820df191fb74bb4c161cb34c6478
    def get_inputs(self):
        return [
            paddle.to_tensor(18720, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_49ed7fd1dd416a2556d890435285925c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7b3da5ca116a3c8cd2602d1697852c4
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_4aa3e3fe1ba475fc4c5c84b6074429c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 9261, 68], dtype='int32'),
        ]


class TestPrimitiveOp_4dea6ee03222c5749434d7540ea3902f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bf33482bd49195c5fe8c0182d505424
    def get_inputs(self):
        return [
            paddle.uniform([4680, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1a2816a259d2a562eb0a2568b9bfe93b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0038e1b2949050eeb1c879e5d3c72857
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[4680, 4], dtype='int64'),
        ]


class PrimitiveOp_34edc944926fad1beb203880b60b3548(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_968983cc2d72c71ab8cf34af13e715dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34edc944926fad1beb203880b60b3548
    def get_inputs(self):
        return [
            paddle.to_tensor(4, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_7fe9d49f358a8f6c92d330f60d8b5c0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34edc944926fad1beb203880b60b3548
    def get_inputs(self):
        return [
            paddle.to_tensor(7, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_1d29f0b94456689b651ffc6140f864ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05c68fcefee09b367cab4d626ad01c5e
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_beac01e30f9900a4e998e2d0879a0835(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 7581, 4], dtype='int32'),
        ]


class TestPrimitiveOp_ee56ea1fdc1710aceabb1237d42c29e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e0d820df191fb74bb4c161cb34c6478
    def get_inputs(self):
        return [
            paddle.to_tensor(15112, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_605eab0ef618bc0c96603f1db502928b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7b3da5ca116a3c8cd2602d1697852c4
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_b9f9e567cdf0d809026d37aa67428b4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 7581, 68], dtype='int32'),
        ]


class TestPrimitiveOp_74cda98c38625c4a71299129cb19e1aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bf33482bd49195c5fe8c0182d505424
    def get_inputs(self):
        return [
            paddle.uniform([3778, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_89b0ff72ac72611912eae0259e42d7ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0038e1b2949050eeb1c879e5d3c72857
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[3778, 4], dtype='int64'),
        ]


class TestPrimitiveOp_02a17e9403f7e9276712ce586f301f85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4888e346918d3535335f3532e1b10a11
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_777ca9997ca07e198aea31ff87514515(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[36], dtype='int64'),
        ]


class TestPrimitiveOp_777ca9997ca07e198aea31ff87514515(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[36], dtype='int64'),
        ]




if __name__ == '__main__':
    unittest.main()