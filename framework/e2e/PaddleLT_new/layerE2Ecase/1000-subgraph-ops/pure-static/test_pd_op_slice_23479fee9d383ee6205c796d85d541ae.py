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



class PrimitiveOp_8c1b948f72b49798cbd5781ec6b0843b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 43, 24, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8727b67dd73bdef20851c6fefd11454b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c1b948f72b49798cbd5781ec6b0843b
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3aaec63e5cb39a51cd67f9d575513786(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c1b948f72b49798cbd5781ec6b0843b
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_b625734245499b5ce7e9f95a6ca197fa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 10, 4, 100, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4502240c583fd622c7189a2bba811aa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b625734245499b5ce7e9f95a6ca197fa
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 100, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_98e7eafe0180b652b5ec4c0073192e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b625734245499b5ce7e9f95a6ca197fa
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 100, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0de193cee2611ab7e89571910bf91677(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b625734245499b5ce7e9f95a6ca197fa
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 100, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_588a622c55742c3fec07b21ac391467f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 43, 16, 6, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3b5d0422819e00da3b992a612ba1c375(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_588a622c55742c3fec07b21ac391467f
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 16, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_039bdc2eaff0320a5e018ced5c97bb6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_588a622c55742c3fec07b21ac391467f
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 16, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e681cd49a01ea13f8f076251644cfbf0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_588a622c55742c3fec07b21ac391467f
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 16, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_b7bc3b3fdd25f0c0b14e76a9615bcca4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 1, 12, 1025, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_346d8b94ea2c6e799482908598823fb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bc3b3fdd25f0c0b14e76a9615bcca4
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 1025, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_00770366b60fd37220fe23b6b8a7ff39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bc3b3fdd25f0c0b14e76a9615bcca4
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 1025, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6cdd577c8bd10aa983ac025f092e69d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bc3b3fdd25f0c0b14e76a9615bcca4
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 1025, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3b5d0422819e00da3b992a612ba1c375(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_588a622c55742c3fec07b21ac391467f
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 16, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_039bdc2eaff0320a5e018ced5c97bb6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_588a622c55742c3fec07b21ac391467f
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 16, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e681cd49a01ea13f8f076251644cfbf0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_588a622c55742c3fec07b21ac391467f
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 16, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_9480975e26ee7b6696ccdaef3f7f0235(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5521f756890417fa815929b91ef45c9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9480975e26ee7b6696ccdaef3f7f0235
    def get_inputs(self):
        return [
            paddle.to_tensor([1542, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cab3a12bcf42233d55953e502e6b0ff3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9480975e26ee7b6696ccdaef3f7f0235
    def get_inputs(self):
        return [
            paddle.to_tensor([1542, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5220613b8d9c4700532587d87561bcbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9480975e26ee7b6696ccdaef3f7f0235
    def get_inputs(self):
        return [
            paddle.to_tensor([2361, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b2278148ae8298c9e29b8b24b31fa49d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9480975e26ee7b6696ccdaef3f7f0235
    def get_inputs(self):
        return [
            paddle.to_tensor([2361, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_924fb76d93eb155871ff97412faef4bb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 11, 16, 6, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a2b09cb6852eb65f7847e532cdec934d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_924fb76d93eb155871ff97412faef4bb
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 16, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_897e43e12d2cf91ca1d80a70d9a547f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_924fb76d93eb155871ff97412faef4bb
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 16, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_40ee042dcadb0df823626b773994f93e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_924fb76d93eb155871ff97412faef4bb
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 16, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c7b972fb687cfdc1fd8843cf56a7c6bb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 1, 8, 1024, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d1206d7cfc4cff89dac45c69409dcbad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7b972fb687cfdc1fd8843cf56a7c6bb
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 1024, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_41ad3e4dd2f25770ef6a588e5887f0fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7b972fb687cfdc1fd8843cf56a7c6bb
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 1024, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_8fb1a0239b0d9299a9e0cc4a8c2e7cff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 100, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_de6815d7089ec4a90cfaa323ece4233f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8fb1a0239b0d9299a9e0cc4a8c2e7cff
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d5b8a90f0a4c949cba9761411a8fa6b2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 54, 3, 197, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_05443e1dee52b7a237760eefebd85b81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5b8a90f0a4c949cba9761411a8fa6b2
    def get_inputs(self):
        return [
            paddle.uniform([3, 54, 3, 197, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_143097c90f13f7640389157a315f5490(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5b8a90f0a4c949cba9761411a8fa6b2
    def get_inputs(self):
        return [
            paddle.uniform([3, 54, 3, 197, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1fc441f4b598d15450f0c463ead64bc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5b8a90f0a4c949cba9761411a8fa6b2
    def get_inputs(self):
        return [
            paddle.uniform([3, 54, 3, 197, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_002dadd7482422713631be80d814bd15(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_539bceb5976f582bc4542f8e0d5da9f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0eb8999fe192516e6a4d5c6454358d23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6d965c48775d0c1bfa80d2ae1ad7ca2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1562426d295650f52accd0374827bc8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0d11184a47ec0b7718eabbb66a02bf57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2dfa13a1e6125c30c09ae099398d874e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fa772800363ec4ad87d1d71fe0fc2b95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0bf296856e13591b4440646a2e1b2511(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_baff407cf6d54fd0eb0b3df2417ebad3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9e95d5afe61e5b7d6312874b5ee3f1af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f877c5035f51ce2dc56837d2f719aaad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_327de8021e9b079ae7cd5c784e900ae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1317a422cdf830e34e39eba1f8b70528(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_58e28abcece3dc92460691334680c8b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a38c4b35ba0773c6e798e904473f5abd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a5b73754c61e60200ea3de286cd4c933(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_57867de638894ff7396e3617096d80bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
            paddle.to_tensor([17], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7d3ebb820c7eda24580d5e45ded178e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([17], dtype='int64').reshape([1]),
            paddle.to_tensor([18], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_23fa32dcf8fbd547209c2011b8fcd24b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([18], dtype='int64').reshape([1]),
            paddle.to_tensor([19], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_03b4795e62141424b19a9984227a43d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([19], dtype='int64').reshape([1]),
            paddle.to_tensor([20], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4aae99902cc64f4b63f142656a6d0572(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([20], dtype='int64').reshape([1]),
            paddle.to_tensor([21], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1cb2cf950ce102fe83837ca1aa6972a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([21], dtype='int64').reshape([1]),
            paddle.to_tensor([22], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_582d319af58d1360862db7d90276cfe8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([22], dtype='int64').reshape([1]),
            paddle.to_tensor([23], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f35c193565434d8611ca7fa4a517d0ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([23], dtype='int64').reshape([1]),
            paddle.to_tensor([24], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4c8d16bd14470da01e20e30f1726f4fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([24], dtype='int64').reshape([1]),
            paddle.to_tensor([25], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f6b5ba25a43e63f52e1a272bf0423e9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([25], dtype='int64').reshape([1]),
            paddle.to_tensor([26], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1e53da74cc902490490adf3391da8e48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([26], dtype='int64').reshape([1]),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b83887ba75275bc7cbadd380a3107ee1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
            paddle.to_tensor([28], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_95f95d3a0c9c31918c2098ac025fbf79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([28], dtype='int64').reshape([1]),
            paddle.to_tensor([29], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_16fdb00dcb2f448d9bcd3a9d6421e390(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([29], dtype='int64').reshape([1]),
            paddle.to_tensor([30], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4d664e607479c38479a5be6d057a675c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([30], dtype='int64').reshape([1]),
            paddle.to_tensor([31], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_99f966b1cf54cb3780654eefbbd53693(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([31], dtype='int64').reshape([1]),
            paddle.to_tensor([32], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_949289366844ec3d5c949d1a929caaa5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([32], dtype='int64').reshape([1]),
            paddle.to_tensor([33], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ae4e7924581c8bbb8b022ae86073c9b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([33], dtype='int64').reshape([1]),
            paddle.to_tensor([34], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_03a0200e51abdac65d339fdcd7bfd802(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([34], dtype='int64').reshape([1]),
            paddle.to_tensor([35], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a91c323b2dfded485dadc0abfd125bbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([35], dtype='int64').reshape([1]),
            paddle.to_tensor([36], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b642a430cf12dda956a2a4275a47554d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([36], dtype='int64').reshape([1]),
            paddle.to_tensor([37], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f1cedc1d2eb2b0cd3118c8b31518f684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([37], dtype='int64').reshape([1]),
            paddle.to_tensor([38], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ad6d7e6ff3ee9ff58963223252960665(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([38], dtype='int64').reshape([1]),
            paddle.to_tensor([39], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_56f1d5d74a99d9d00f9203109c2ac49d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([39], dtype='int64').reshape([1]),
            paddle.to_tensor([40], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6183ea61b77877d966f0d8061f37dd73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([40], dtype='int64').reshape([1]),
            paddle.to_tensor([41], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3eb8a2d880610c2c705cbe571d649afe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([41], dtype='int64').reshape([1]),
            paddle.to_tensor([42], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_696815cc19cb66610afac9df24e2b3c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([42], dtype='int64').reshape([1]),
            paddle.to_tensor([43], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_698609ef346d3d77c783a9e769f9ebb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([43], dtype='int64').reshape([1]),
            paddle.to_tensor([44], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9c26a8fdc7c2a770490650c61069721b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([44], dtype='int64').reshape([1]),
            paddle.to_tensor([45], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_eadc4003308a2263aeca1568763f36ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([45], dtype='int64').reshape([1]),
            paddle.to_tensor([46], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2856400cbc71379f7184b1e3db8525c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([46], dtype='int64').reshape([1]),
            paddle.to_tensor([47], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_80ac419df5eaaf3a87bf9cd0a2a24fae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([47], dtype='int64').reshape([1]),
            paddle.to_tensor([48], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_87fd902ae483c3df8e4d2877d23e804a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([48], dtype='int64').reshape([1]),
            paddle.to_tensor([49], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_96bf0c5452dc7be61e25fcbea763f393(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([49], dtype='int64').reshape([1]),
            paddle.to_tensor([50], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c5833c68d7485f1ea35e0ac9a29d855d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([50], dtype='int64').reshape([1]),
            paddle.to_tensor([51], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8c9030c422116ad6ddd3eed15bc720bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([51], dtype='int64').reshape([1]),
            paddle.to_tensor([52], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3cb474ad278b1a3d41daaa9f08e6c834(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([52], dtype='int64').reshape([1]),
            paddle.to_tensor([53], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0b310dacd9f8ce1b9c59617a4095cfc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([53], dtype='int64').reshape([1]),
            paddle.to_tensor([54], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3891f02677659ece9a0935c3b5817217(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([54], dtype='int64').reshape([1]),
            paddle.to_tensor([55], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6a717e337dea4fb619b174e8483c4ac2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([55], dtype='int64').reshape([1]),
            paddle.to_tensor([56], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4410c87e36a951ddc3a6d07b41544925(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([56], dtype='int64').reshape([1]),
            paddle.to_tensor([57], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5fee97c4c679a9ac6aaed9d09cf9356a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([57], dtype='int64').reshape([1]),
            paddle.to_tensor([58], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3fb52e9030de311a2e0223779b48de93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([58], dtype='int64').reshape([1]),
            paddle.to_tensor([59], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_92c5424f46d0d25e6ce22bba000cb914(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([59], dtype='int64').reshape([1]),
            paddle.to_tensor([60], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d12d47f3e8f341d836134195a5d8b4da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([60], dtype='int64').reshape([1]),
            paddle.to_tensor([61], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4c8029475d86b3810af1065da599a8ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([61], dtype='int64').reshape([1]),
            paddle.to_tensor([62], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e705af985a2dc1e3ae3e95ba491361b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([62], dtype='int64').reshape([1]),
            paddle.to_tensor([63], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_60ab58482e9b1b87bc1521663c75fdcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([63], dtype='int64').reshape([1]),
            paddle.to_tensor([64], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5134152164c6876ef6d340e886078e2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([64], dtype='int64').reshape([1]),
            paddle.to_tensor([65], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d1b19f2620f574422c5f27dafbfca343(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([65], dtype='int64').reshape([1]),
            paddle.to_tensor([66], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7687e4fd68b9008a0acb225b54f950fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([66], dtype='int64').reshape([1]),
            paddle.to_tensor([67], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2a495976d5775dc1d0cd5c1817a624c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([67], dtype='int64').reshape([1]),
            paddle.to_tensor([68], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1ab8ef56f441811781fa286d5d6becec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([68], dtype='int64').reshape([1]),
            paddle.to_tensor([69], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_385d9cb12b25acc50b827bf066bd0532(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([69], dtype='int64').reshape([1]),
            paddle.to_tensor([70], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b1b88b035bd0162ec5f81aaaf292deb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([70], dtype='int64').reshape([1]),
            paddle.to_tensor([71], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e093d27418e939e2d8b2a1d6efbfa3bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([71], dtype='int64').reshape([1]),
            paddle.to_tensor([72], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3d9daaabce620135d350b3e6b1d8085f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([72], dtype='int64').reshape([1]),
            paddle.to_tensor([73], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_139da6832a118dbabcd36eb1eea658e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([73], dtype='int64').reshape([1]),
            paddle.to_tensor([74], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd0d9be988a2c9bbd34ee95e174e336b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([74], dtype='int64').reshape([1]),
            paddle.to_tensor([75], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c09514a72175f866e26248565dd5640d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([75], dtype='int64').reshape([1]),
            paddle.to_tensor([76], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_914194c109f3422c7b676c0d098b4280(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([76], dtype='int64').reshape([1]),
            paddle.to_tensor([77], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d85f4f299f1fa66a36109301ea0ad6f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([77], dtype='int64').reshape([1]),
            paddle.to_tensor([78], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bb47051014bd18f9b5c28f18dd8b4e82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([78], dtype='int64').reshape([1]),
            paddle.to_tensor([79], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_69a7c79adab9693cc7d818e98b56d9ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([79], dtype='int64').reshape([1]),
            paddle.to_tensor([80], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_44015bb8945362d3bc4b0f60ef25a4a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([80], dtype='int64').reshape([1]),
            paddle.to_tensor([81], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c23166e67c65597edb0b2e1381abafd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([81], dtype='int64').reshape([1]),
            paddle.to_tensor([82], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_62e506dcdb63cbb75b46c79c368a46b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([82], dtype='int64').reshape([1]),
            paddle.to_tensor([83], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6f0bc68b2f1d2a616ae1797e73a33cbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([83], dtype='int64').reshape([1]),
            paddle.to_tensor([84], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_963a3933a75f7b9a1fc4ae13be862995(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([84], dtype='int64').reshape([1]),
            paddle.to_tensor([85], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_33f8f7322608e4205967e7a106b6694e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([85], dtype='int64').reshape([1]),
            paddle.to_tensor([86], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5a5c8a4ac08e4c6be47c8ba333d49bcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([86], dtype='int64').reshape([1]),
            paddle.to_tensor([87], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_520c505e9cbb7589482231d774b86f35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([87], dtype='int64').reshape([1]),
            paddle.to_tensor([88], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8bde9768b980697ad832f0af90d92932(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([88], dtype='int64').reshape([1]),
            paddle.to_tensor([89], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6c2dd0a9cea2fe8a7528234d2ab4c11a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([89], dtype='int64').reshape([1]),
            paddle.to_tensor([90], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b75ceb02b9bf0eab146abd1426ae82d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([90], dtype='int64').reshape([1]),
            paddle.to_tensor([91], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bf3e46e1f63d69c31908527c891f3cf6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([91], dtype='int64').reshape([1]),
            paddle.to_tensor([92], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_75f02883fa5a442f4082689622e42509(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([92], dtype='int64').reshape([1]),
            paddle.to_tensor([93], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9c329e54b764fe107d644a7a37d082c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([93], dtype='int64').reshape([1]),
            paddle.to_tensor([94], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f5dafac3b1e999e50464666efd9a5f93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([94], dtype='int64').reshape([1]),
            paddle.to_tensor([95], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2d6e7e77d84cfeff3780b1037f6afbbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([95], dtype='int64').reshape([1]),
            paddle.to_tensor([96], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f40611e2b22b53ec99d626386ee8d67b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([96], dtype='int64').reshape([1]),
            paddle.to_tensor([97], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9048bc94df2baa008797aa09cd871976(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([97], dtype='int64').reshape([1]),
            paddle.to_tensor([98], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1f2bb9740be3b4f8b8bee9bb74252406(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([98], dtype='int64').reshape([1]),
            paddle.to_tensor([99], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_42a661283d4e1fef54b025d95efed282(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([99], dtype='int64').reshape([1]),
            paddle.to_tensor([100], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_926e58f784a0e313d98c4365523dbfd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([100], dtype='int64').reshape([1]),
            paddle.to_tensor([101], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_314100e225d758d6ac6a5dad18fb3769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([101], dtype='int64').reshape([1]),
            paddle.to_tensor([102], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8114c6fd5e8a1f377b9d042a224d3a21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([102], dtype='int64').reshape([1]),
            paddle.to_tensor([103], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_53ead92fc89d154fcaddb5f5037e006b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([103], dtype='int64').reshape([1]),
            paddle.to_tensor([104], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fa30249903bcad9d36e8102f7189f4b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([104], dtype='int64').reshape([1]),
            paddle.to_tensor([105], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d7fcdab27d7464ee45fefe09f2fd0b2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([105], dtype='int64').reshape([1]),
            paddle.to_tensor([106], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9219868b4f14816e354e0d0e765642f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([106], dtype='int64').reshape([1]),
            paddle.to_tensor([107], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3b344fb430dc7ef1f01dd2e63798a555(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([107], dtype='int64').reshape([1]),
            paddle.to_tensor([108], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f8963abfa87724d855ee2e9ccb1f1b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([108], dtype='int64').reshape([1]),
            paddle.to_tensor([109], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4225670beda403c85db2e244c89efe5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([109], dtype='int64').reshape([1]),
            paddle.to_tensor([110], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_965898ec5e977d63cb9140cdea6db90d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([110], dtype='int64').reshape([1]),
            paddle.to_tensor([111], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3f25e8fddb92c690cf69e6c3f38834fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([111], dtype='int64').reshape([1]),
            paddle.to_tensor([112], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f4377c196a42cb96df83462c965a7e5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([112], dtype='int64').reshape([1]),
            paddle.to_tensor([113], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4ebc8f3907228a713a537202dc47e077(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([113], dtype='int64').reshape([1]),
            paddle.to_tensor([114], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1b4fcbca0ef34316b13c1b9b6d2a7cf0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([114], dtype='int64').reshape([1]),
            paddle.to_tensor([115], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e4fe53661ce7fb57c1351b358b09c759(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([115], dtype='int64').reshape([1]),
            paddle.to_tensor([116], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ff7b0bb6d30d47c0104aaa1ffb50ff6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([116], dtype='int64').reshape([1]),
            paddle.to_tensor([117], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4084a1ad19d1538b376de948fbe802fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([117], dtype='int64').reshape([1]),
            paddle.to_tensor([118], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7eb33f2185e92c31ffd8a7711bd0ccf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([118], dtype='int64').reshape([1]),
            paddle.to_tensor([119], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a273e9a169e946aa58b9174eff0fab9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([119], dtype='int64').reshape([1]),
            paddle.to_tensor([120], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_95f777d8e5fba8a87020a1a5e55b9f7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([120], dtype='int64').reshape([1]),
            paddle.to_tensor([121], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c936ad1b7be07230629c9e3fceb30afa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([121], dtype='int64').reshape([1]),
            paddle.to_tensor([122], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_636d1c7c63a74a59f52e0dd8abee7f4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([122], dtype='int64').reshape([1]),
            paddle.to_tensor([123], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_15afe814b480ab821e8bd3858ccd1fc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([123], dtype='int64').reshape([1]),
            paddle.to_tensor([124], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1316e0cb12a0f534d2e39694ff085002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([124], dtype='int64').reshape([1]),
            paddle.to_tensor([125], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_108954927ae7aaa29fc5e5b961bba952(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([125], dtype='int64').reshape([1]),
            paddle.to_tensor([126], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_83d15e969aee39971d4c89c2e12080a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([126], dtype='int64').reshape([1]),
            paddle.to_tensor([127], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_70c1b91b440976b580f4b243d8fe7546(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([127], dtype='int64').reshape([1]),
            paddle.to_tensor([128], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_525a564acb9a41c01aa7174a520ca890(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([128], dtype='int64').reshape([1]),
            paddle.to_tensor([129], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_609af81deb7d72484a61c801362d648c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([129], dtype='int64').reshape([1]),
            paddle.to_tensor([130], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4c2abe7434af75d2e65cff88e5aaac8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([130], dtype='int64').reshape([1]),
            paddle.to_tensor([131], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2abe58d5dcaed9348c36e7ccf73a33aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([131], dtype='int64').reshape([1]),
            paddle.to_tensor([132], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bafe30fed0dafb7a61bde3816bc9aab3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([132], dtype='int64').reshape([1]),
            paddle.to_tensor([133], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_22c9967eac7c1dcd5d1530df3fc3d251(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([133], dtype='int64').reshape([1]),
            paddle.to_tensor([134], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_02534570b4090748bc00eb58fc7bcdc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([134], dtype='int64').reshape([1]),
            paddle.to_tensor([135], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7a05d6e9c41faec63c4f3ef08a60a901(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([135], dtype='int64').reshape([1]),
            paddle.to_tensor([136], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_078316dd79afeb9b33032d1f823fab1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([136], dtype='int64').reshape([1]),
            paddle.to_tensor([137], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_70bed2fe4fab0c20e89ba5e678ab9fae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([137], dtype='int64').reshape([1]),
            paddle.to_tensor([138], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1a863425cf2ebf2c08d67d4bb9de3587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([138], dtype='int64').reshape([1]),
            paddle.to_tensor([139], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1eb2853169c9838c0da7142a6ce52eba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([139], dtype='int64').reshape([1]),
            paddle.to_tensor([140], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f258ad89342a13ac5b3136f828fb1054(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([140], dtype='int64').reshape([1]),
            paddle.to_tensor([141], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0ef11e5b01bf6ab5efaae7d49acf7edc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([141], dtype='int64').reshape([1]),
            paddle.to_tensor([142], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_442fe785858d76b4a98029bd3fce7424(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([142], dtype='int64').reshape([1]),
            paddle.to_tensor([143], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_91650a5c4a4ce3603059f39434b78bb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([143], dtype='int64').reshape([1]),
            paddle.to_tensor([144], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f2ae7903ca8e0dc0ac610f3cbffd106b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([144], dtype='int64').reshape([1]),
            paddle.to_tensor([145], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_35917b2ae5a4dc44ed1690757d0e4798(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([145], dtype='int64').reshape([1]),
            paddle.to_tensor([146], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_eb96c48b191fa46a91e98922cddb668e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([146], dtype='int64').reshape([1]),
            paddle.to_tensor([147], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4d57cd55920e5c773d7631fb15224ffe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([147], dtype='int64').reshape([1]),
            paddle.to_tensor([148], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7d05c8948050ab028363aff8eaa9200e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([148], dtype='int64').reshape([1]),
            paddle.to_tensor([149], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_899e302b34d4a8323f383218e73ee0b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([149], dtype='int64').reshape([1]),
            paddle.to_tensor([150], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a3ddf90dd9694016648ef142c33544a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([150], dtype='int64').reshape([1]),
            paddle.to_tensor([151], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_13022e7659b3d29af6d380de2969ffc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([151], dtype='int64').reshape([1]),
            paddle.to_tensor([152], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ede2bf20994abe201043d4a865643516(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([152], dtype='int64').reshape([1]),
            paddle.to_tensor([153], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_80986b33d93bf2d7718694a6157c2a29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([153], dtype='int64').reshape([1]),
            paddle.to_tensor([154], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fc743dd385adc040ed74de2f3f61efec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([154], dtype='int64').reshape([1]),
            paddle.to_tensor([155], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7b7b5c3bf1de9ac0f65a6a90f4643f4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([155], dtype='int64').reshape([1]),
            paddle.to_tensor([156], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_383a20e1032483dea112fee03ab1b372(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([156], dtype='int64').reshape([1]),
            paddle.to_tensor([157], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2ea26c632e28502701b6cc468741f68c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([157], dtype='int64').reshape([1]),
            paddle.to_tensor([158], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1ed1dd2f44714fee168ca867711b2366(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([158], dtype='int64').reshape([1]),
            paddle.to_tensor([159], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f53931611e6ddfccae8f7a460bda64b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([159], dtype='int64').reshape([1]),
            paddle.to_tensor([160], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6b85c3604fcbc36c63707e88159a8262(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([160], dtype='int64').reshape([1]),
            paddle.to_tensor([161], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5302cf3fdb3c9613a69186b1d376680f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([161], dtype='int64').reshape([1]),
            paddle.to_tensor([162], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d86e13166419f6bcc00affe6366ea65a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([162], dtype='int64').reshape([1]),
            paddle.to_tensor([163], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5148803cdb1fa25deafdaa52b86588e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([163], dtype='int64').reshape([1]),
            paddle.to_tensor([164], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_afb42e47053ac0975c6080feabdf8406(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([164], dtype='int64').reshape([1]),
            paddle.to_tensor([165], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_71d449439a92cee78e197aa1d8e420c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([165], dtype='int64').reshape([1]),
            paddle.to_tensor([166], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_658c1f3649e08ab71e5128b48f62a495(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([166], dtype='int64').reshape([1]),
            paddle.to_tensor([167], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cc9241ded916095df68f1992f598819f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([167], dtype='int64').reshape([1]),
            paddle.to_tensor([168], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ce48e5913839e8e2d90464de383c2c45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([168], dtype='int64').reshape([1]),
            paddle.to_tensor([169], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c6cd382c899232b095c533358109423e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([169], dtype='int64').reshape([1]),
            paddle.to_tensor([170], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b2ed207b771acec078963863853f9da3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([170], dtype='int64').reshape([1]),
            paddle.to_tensor([171], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_75fc58c8c25dc930122cd78a5308455c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([171], dtype='int64').reshape([1]),
            paddle.to_tensor([172], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_36571bae712f6d1a0c53534e90460915(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([172], dtype='int64').reshape([1]),
            paddle.to_tensor([173], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c3fcab4fe774cd94d4b60362e0d32874(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([173], dtype='int64').reshape([1]),
            paddle.to_tensor([174], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_eed2ade3479ae5aa0e2263338f00896d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([174], dtype='int64').reshape([1]),
            paddle.to_tensor([175], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_88fd4c7439aa479f865d5d14a450f8e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([175], dtype='int64').reshape([1]),
            paddle.to_tensor([176], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fc63e24600f3151995f65ac1cd63158b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([176], dtype='int64').reshape([1]),
            paddle.to_tensor([177], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7e58e8bea9674ff562fbd6f471452d55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([177], dtype='int64').reshape([1]),
            paddle.to_tensor([178], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bff72a20ba33effc1b0bd5477a8d429a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([178], dtype='int64').reshape([1]),
            paddle.to_tensor([179], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b826c3b3ca2f66a7540bc022c7eb93d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([179], dtype='int64').reshape([1]),
            paddle.to_tensor([180], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_726cacc94973f17fb1b013e6c97a4e9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([180], dtype='int64').reshape([1]),
            paddle.to_tensor([181], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_725305d61771275999acd506b62c40ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([181], dtype='int64').reshape([1]),
            paddle.to_tensor([182], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_da3e66a15dd90d0ecb55d4ef56f9dfc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([182], dtype='int64').reshape([1]),
            paddle.to_tensor([183], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9bb620afc794ee39b083412903afb4f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([183], dtype='int64').reshape([1]),
            paddle.to_tensor([184], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_42c53eecd0ba3cc3eb9693255d1286d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([184], dtype='int64').reshape([1]),
            paddle.to_tensor([185], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_08d2b46568f75e132b2b7d03863a615e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([185], dtype='int64').reshape([1]),
            paddle.to_tensor([186], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fb983560a92073ee1db3a72c29a2b479(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([186], dtype='int64').reshape([1]),
            paddle.to_tensor([187], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_78b4e617282d9d76b1aefd18800393e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([187], dtype='int64').reshape([1]),
            paddle.to_tensor([188], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6eb09c26493cbcb447ff02e50621b99c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([188], dtype='int64').reshape([1]),
            paddle.to_tensor([189], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ba05ee38641d3f6dfada0c860a343612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([189], dtype='int64').reshape([1]),
            paddle.to_tensor([190], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_17448e517b70b93eb1a0443d3d42a1fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([190], dtype='int64').reshape([1]),
            paddle.to_tensor([191], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8b2df81aa14acef1227092ef59839432(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([191], dtype='int64').reshape([1]),
            paddle.to_tensor([192], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_72e24c45b28eb5b889904ba2cf81da43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([192], dtype='int64').reshape([1]),
            paddle.to_tensor([193], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cf5499cd7fcf2532841d1e1a4bdf68c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([193], dtype='int64').reshape([1]),
            paddle.to_tensor([194], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3619f2bcc35d32102cbaa0d462f4e5e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([194], dtype='int64').reshape([1]),
            paddle.to_tensor([195], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_69ee20652875c6a13a7369f2d459d87b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002dadd7482422713631be80d814bd15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([195], dtype='int64').reshape([1]),
            paddle.to_tensor([196], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6f1169f9071ce5743e24091fa800b817(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4ecb9138f0021b6edb980298b58d8397(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0238a9b7251f0d40866cdf80199c02f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d55aaaa33adefb17f34b84c755d178f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 43, 4, 12, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ed6b73f16e18794bb6d6092a3df560f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d55aaaa33adefb17f34b84c755d178f9
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 4, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_729e8bdc01d10cec4681e78fe1b8eea8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d55aaaa33adefb17f34b84c755d178f9
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 4, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_38f0b858a1e75dae3ebdfea38bdd883f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d55aaaa33adefb17f34b84c755d178f9
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 4, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ff1b21f6c7c7c514facff0a4cd1b5e88(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 11, 4, 12, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7dc5b880df3a50b020226673d300062f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff1b21f6c7c7c514facff0a4cd1b5e88
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 4, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b6160058404152c6c763aeb91212d85a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff1b21f6c7c7c514facff0a4cd1b5e88
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 4, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0520811acb435fb32f24030febc35672(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff1b21f6c7c7c514facff0a4cd1b5e88
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 4, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_db37ee58672f6b1d44a4ee9bd0d79e52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8423b9183a7d5ac6f7b7a95f9e904e0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_cba436843473132c18085e1a14d5ef35(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 43, 6, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a3e92ad44e8132a9bf60cfe54d10a900(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cba436843473132c18085e1a14d5ef35
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47895a093b63a5c4b886d2750d01b220(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cba436843473132c18085e1a14d5ef35
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ca02d8e37b6b957e64ec7ad7a1e91ff0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 64, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_795ae4e5d9cf7e901c92fdc1b0c10fdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 64, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_7f1d83217d5c295dcd1a3d6ea0e3da7c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 10, 2, 640, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_088906b4ab7d59c24b9afaa09cdb1a27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f1d83217d5c295dcd1a3d6ea0e3da7c
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 2, 640, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dae720186611b2f44cd0bd2dfc41647a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f1d83217d5c295dcd1a3d6ea0e3da7c
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 2, 640, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bf6ca7f6d0a5dbad837a465f29c243eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f1d83217d5c295dcd1a3d6ea0e3da7c
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 2, 640, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a3e92ad44e8132a9bf60cfe54d10a900(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cba436843473132c18085e1a14d5ef35
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47895a093b63a5c4b886d2750d01b220(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cba436843473132c18085e1a14d5ef35
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4f063157bd604da0ae2ae05c617410ad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 43, 64, 3, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7dfdb5548be93c122f9202a91dc42b3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f063157bd604da0ae2ae05c617410ad
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 64, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c5b4868a4f5994a0f46251792619c74a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f063157bd604da0ae2ae05c617410ad
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 64, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0b19c74abc0fa4e9205f5ac551c62acd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f063157bd604da0ae2ae05c617410ad
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 64, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d8ea3acb02494632235f86136968867d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 10, 2, 200, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_373b8e275b25ea92d58bc453f264974c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8ea3acb02494632235f86136968867d
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 2, 200, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2437015b055d319d632b0a67c8956617(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8ea3acb02494632235f86136968867d
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 2, 200, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_803680bb9ff36ebd4308134f7ddbbbad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8ea3acb02494632235f86136968867d
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 2, 200, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_829887b03bde77c51b681ba316d3074c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 43, 12, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d5bbb94176f6740783f7afa32eae67e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_829887b03bde77c51b681ba316d3074c
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7cb94362bc257b970836f8e526f4507c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_829887b03bde77c51b681ba316d3074c
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ca02d8e37b6b957e64ec7ad7a1e91ff0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 64, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_795ae4e5d9cf7e901c92fdc1b0c10fdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 64, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_123390e893014beda794837cbe8ff1ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fcd17fa1e46e787b3f84ee62e6cd2b6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d8f87f669abcdb86c1b01768647cb8fe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 11, 6, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f8abb13541a7b6fac023b092f86f9891(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8f87f669abcdb86c1b01768647cb8fe
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4fe913b85b951b8e10d0f16e58dc5198(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8f87f669abcdb86c1b01768647cb8fe
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_2be559ce424338711c1cbec0c7d16111(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 11, 3, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1b2dc6f1e3f8a1a8d27fe0eba724169a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2be559ce424338711c1cbec0c7d16111
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_32213ecef66619d3d4d981319a22bbea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2be559ce424338711c1cbec0c7d16111
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4bfb6d392558691a4598f8521a6a42d3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 4312, 4, 16, 6], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1a135b43b7511f34dce498d54a1f292e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bfb6d392558691a4598f8521a6a42d3
    def get_inputs(self):
        return [
            paddle.uniform([2, 4312, 4, 16, 6], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_52ce504cf88a599c5695baf9cbdf8c76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bfb6d392558691a4598f8521a6a42d3
    def get_inputs(self):
        return [
            paddle.uniform([2, 4312, 4, 16, 6], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_e1b2bf71982cdba2694a56be835df15a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 11, 1, 24, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ddc5e38b09ab591fb6c9af49c7ffdbdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1b2bf71982cdba2694a56be835df15a
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6886b4652668c7f30e161fe3d529e7c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1b2bf71982cdba2694a56be835df15a
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c11e1dcbdf4c29e8baefd51f66c5e9be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1b2bf71982cdba2694a56be835df15a
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_b1f412f03d27b0c1f386d736846ac8d6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 300, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5e72c2304040a99d8e19c2bf5c16b6f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1f412f03d27b0c1f386d736846ac8d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_25c9b174fa4874bf8e39115e8d94ebeb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 10, 8, 160, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_487f288bb20ca0c8eb4505fca3b81d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25c9b174fa4874bf8e39115e8d94ebeb
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 8, 160, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04bd649664dc6c4abb5a4bc21a6d7ab1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25c9b174fa4874bf8e39115e8d94ebeb
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 8, 160, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fd2a94eebe05672ce82d30e228c7c136(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25c9b174fa4874bf8e39115e8d94ebeb
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 8, 160, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5e72c2304040a99d8e19c2bf5c16b6f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1f412f03d27b0c1f386d736846ac8d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d8a6e4d9ef5534f98e9490697d754869(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9480975e26ee7b6696ccdaef3f7f0235
    def get_inputs(self):
        return [
            paddle.to_tensor([2053, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f74dcbc3dd65e1946784153ee3fad8f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9480975e26ee7b6696ccdaef3f7f0235
    def get_inputs(self):
        return [
            paddle.to_tensor([2053, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a2b09cb6852eb65f7847e532cdec934d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_924fb76d93eb155871ff97412faef4bb
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 16, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_897e43e12d2cf91ca1d80a70d9a547f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_924fb76d93eb155871ff97412faef4bb
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 16, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_40ee042dcadb0df823626b773994f93e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_924fb76d93eb155871ff97412faef4bb
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 16, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f5560710b9ffa50e7baaf4b1dcfb6b1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2fe184c3c7f3b493a1d1ceb6d5011ea7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ea5c52a37ea3ebdf749f0c7de615e713(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 10, 4, 320, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0f8786484e75f1745c7e58e81bb89f5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea5c52a37ea3ebdf749f0c7de615e713
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 320, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_383d16f35ce04a7a8f4da7a2d04bcfe5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea5c52a37ea3ebdf749f0c7de615e713
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 320, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e401816b5f61fefedddfbd553b921db3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea5c52a37ea3ebdf749f0c7de615e713
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 320, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_69e5d755124f1277f4d2e2b2be493eba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_551749f4ea5c6083d04e62d29cd591ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5983a17160f158007273d1f883b2f74b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6a93e024de2a2e826b89c6646dced73a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5aea6d2bc70452077dfae3d6ccf60472(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e0fa3ad8fbe731724744649ac3ec8a4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b5e0f88127cf042d36a3665a0493c89f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_167c14a7cd7d5cad77ba460ae0a18f87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2a9a97c1cc4347612d2bad0ca1db7b48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6af549887e3aaafb2587fa3c3cbb7807(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_669f8530b3a8c8dd82693dcdf41894a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d1639cbad5a76dff9c01d44f9614b023(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_406275f756e9d76c5fc39c55772b13d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4a03b08fad380b94831f43ffbfa07f01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5e0d963a0d58b45620dce9dcf3f8c3c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6b4796e79c3361704da87f3ca9fed936(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d9a8635a10b4c26a930a8fc4f85ebcce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
            paddle.to_tensor([17], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_221ccf66a6587aa204c7ffc2bec3751a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([17], dtype='int64').reshape([1]),
            paddle.to_tensor([18], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_be9d4ef7af10c9ebae89f3292b6484e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([18], dtype='int64').reshape([1]),
            paddle.to_tensor([19], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_94499bab628cb9596352246b519c1266(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([19], dtype='int64').reshape([1]),
            paddle.to_tensor([20], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_15cdb767c5ef2132c1af4471f5c9c45e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([20], dtype='int64').reshape([1]),
            paddle.to_tensor([21], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0fe054bc658f07885627189ad094b059(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([21], dtype='int64').reshape([1]),
            paddle.to_tensor([22], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b476a0a14489321ac281297339bf99cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([22], dtype='int64').reshape([1]),
            paddle.to_tensor([23], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_80d03fa53dc23944eb7787547525daa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([23], dtype='int64').reshape([1]),
            paddle.to_tensor([24], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f92a44f7f8967f91ff2047c77e411f79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([24], dtype='int64').reshape([1]),
            paddle.to_tensor([25], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_69b2d00c85a48041e978b7fa4937f34a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([25], dtype='int64').reshape([1]),
            paddle.to_tensor([26], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1a7eb00a83da6dd59ea515f094fe2f66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([26], dtype='int64').reshape([1]),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a4a3f1a82a5662537ebf7465510c0318(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
            paddle.to_tensor([28], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5a1fcc6d4ffc24b1a761e55248df030b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([28], dtype='int64').reshape([1]),
            paddle.to_tensor([29], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_87a01ff81054c56372cda80c74b2f7e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([29], dtype='int64').reshape([1]),
            paddle.to_tensor([30], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_718934b35303c9799e56fedaa5ec9841(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([30], dtype='int64').reshape([1]),
            paddle.to_tensor([31], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9b5d9f5bd32423a8274deba4b14a8a68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([31], dtype='int64').reshape([1]),
            paddle.to_tensor([32], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5487ac49a59efde17de5af12e30648a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([32], dtype='int64').reshape([1]),
            paddle.to_tensor([33], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fa3df3a199d58f72c8d3a1ad86b995f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([33], dtype='int64').reshape([1]),
            paddle.to_tensor([34], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e84cdb9d9ead871bffe1e59671e101ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([34], dtype='int64').reshape([1]),
            paddle.to_tensor([35], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_026b9a341dd5411a20a71124faf10c10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([35], dtype='int64').reshape([1]),
            paddle.to_tensor([36], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_798bc2fb6530cce03678f43654f449ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([36], dtype='int64').reshape([1]),
            paddle.to_tensor([37], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b64c72409e0e862cb6e3cc3469cf0d87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([37], dtype='int64').reshape([1]),
            paddle.to_tensor([38], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_87f54a653c31e6f637433508608d1111(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([38], dtype='int64').reshape([1]),
            paddle.to_tensor([39], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c1dc23c75bf9c36edccd4afb0cb7497c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([39], dtype='int64').reshape([1]),
            paddle.to_tensor([40], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_79d15e1eed03fa2ddd41362a7c44ec38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([40], dtype='int64').reshape([1]),
            paddle.to_tensor([41], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_aefdf2ebb2c04bd173846cacd772debb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([41], dtype='int64').reshape([1]),
            paddle.to_tensor([42], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a71177735f5edfe6efef21933cffdbe7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([42], dtype='int64').reshape([1]),
            paddle.to_tensor([43], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_391c2e5642883f364a172c0fc5d1576e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([43], dtype='int64').reshape([1]),
            paddle.to_tensor([44], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3dd61c04c27ef4b0a675c4a95609c4e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([44], dtype='int64').reshape([1]),
            paddle.to_tensor([45], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_319756dbe7fd265186db7f00bf085fee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([45], dtype='int64').reshape([1]),
            paddle.to_tensor([46], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b0a59a6e473745dafb4513c3e477e802(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([46], dtype='int64').reshape([1]),
            paddle.to_tensor([47], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3e2b481d7f183ef7b674f3f8274dfb42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([47], dtype='int64').reshape([1]),
            paddle.to_tensor([48], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a5bd20478a874668aa6983d011d28396(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0f25e71ef23f0a3e2dfb06ed543726
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([48], dtype='int64').reshape([1]),
            paddle.to_tensor([49], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ccbc1decc6660609ba7ad8594bf7d59d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 256, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7cee05b50945d69f7f7ff92a0d64da2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 256, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6c8d1a5a3004379af8aeb63bfc03e801(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16, 16], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d7a1e96e7b4a1ac7f22b52706f1e697e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c8d1a5a3004379af8aeb63bfc03e801
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1710863b71b4e93de2b5998568e0731b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c8d1a5a3004379af8aeb63bfc03e801
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_00a7984d901462600c7ee120273a3be2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c8d1a5a3004379af8aeb63bfc03e801
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_774f59d7cf3a2558490f38b1653d7699(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c8d1a5a3004379af8aeb63bfc03e801
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4b04bba7851216d849e75de318180cf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c8d1a5a3004379af8aeb63bfc03e801
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_42918eee83b3d140447206234f9b0221(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c8d1a5a3004379af8aeb63bfc03e801
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_29fa0717e01669ad3764b9c82539afad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c8d1a5a3004379af8aeb63bfc03e801
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_63f884ec21aa57013a296833d4950476(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c8d1a5a3004379af8aeb63bfc03e801
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_927e065e43403a7233b88e841f51f666(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c8d1a5a3004379af8aeb63bfc03e801
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5de979f41fa8921e7f1d16cec795f5bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c8d1a5a3004379af8aeb63bfc03e801
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ccb76214c4b2f7858726538955b55744(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c8d1a5a3004379af8aeb63bfc03e801
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fed24bb341c4f356f84b2a5c44b1043d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c8d1a5a3004379af8aeb63bfc03e801
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cc8eb9e11b308bfbb2659f7ba1264535(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c8d1a5a3004379af8aeb63bfc03e801
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6fb32ef2d0a3993bd112448bfba56af9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c8d1a5a3004379af8aeb63bfc03e801
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ef0baadc21aa1bdc5dde13ceb2bceee8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c8d1a5a3004379af8aeb63bfc03e801
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6ba9c0bece94ffc5bb64889cc3c15b24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c8d1a5a3004379af8aeb63bfc03e801
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1bb15bc4a85945b62a54abd130ac0098(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9480975e26ee7b6696ccdaef3f7f0235
    def get_inputs(self):
        return [
            paddle.to_tensor([1825, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_28686cd0cd0d7595aac013aaa8765fde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9480975e26ee7b6696ccdaef3f7f0235
    def get_inputs(self):
        return [
            paddle.to_tensor([1825, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4e5312de63c3d4eff8a9cc729eb298ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 1, 8, 512, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_72affd284e73d42b9246c8097044da01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e5312de63c3d4eff8a9cc729eb298ae
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 512, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_df46f2947a58910db37462057c2261b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e5312de63c3d4eff8a9cc729eb298ae
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 512, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4f104df590b121e14c46663c1526d4d6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 54, 3, 198, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_35f37977abebd329306904beabbb2c97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f104df590b121e14c46663c1526d4d6
    def get_inputs(self):
        return [
            paddle.uniform([3, 54, 3, 198, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_301b142cf29b57e1f600922e88cdd9a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f104df590b121e14c46663c1526d4d6
    def get_inputs(self):
        return [
            paddle.uniform([3, 54, 3, 198, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e951b01bcf7718e18c6d7f5c1d3c256e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f104df590b121e14c46663c1526d4d6
    def get_inputs(self):
        return [
            paddle.uniform([3, 54, 3, 198, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c9927c614c2fad25ab802fa0f352724d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9480975e26ee7b6696ccdaef3f7f0235
    def get_inputs(self):
        return [
            paddle.to_tensor([3087, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_122a5b07841b0ab7ed9f39304ac7db03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9480975e26ee7b6696ccdaef3f7f0235
    def get_inputs(self):
        return [
            paddle.to_tensor([3087, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_8d7952889068d6bd7d3c0aa359aeaf3f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 1, 8, 1024, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2aecb463a292fac147177fbbb1da0542(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d7952889068d6bd7d3c0aa359aeaf3f
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 1024, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_504aa92129d954f64c6676c1391b3126(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d7952889068d6bd7d3c0aa359aeaf3f
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 1024, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d5bbb94176f6740783f7afa32eae67e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_829887b03bde77c51b681ba316d3074c
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7cb94362bc257b970836f8e526f4507c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_829887b03bde77c51b681ba316d3074c
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_0d3785f190e39e98fa47ec1e321da0e4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 43, 3, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e1daa0bc7b4a96205b4b16551de428ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d3785f190e39e98fa47ec1e321da0e4
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a66b7d09191c01b7055974b2e2bd56e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d3785f190e39e98fa47ec1e321da0e4
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_123f4206f30e0c7376b3076b177cb920(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 11, 64, 3, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_39c44bfb339def15ece36528aaf8437d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_123f4206f30e0c7376b3076b177cb920
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 64, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_798fc51913fb164464c70d7e966d5ff3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_123f4206f30e0c7376b3076b177cb920
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 64, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4f3e1dfd705b1190bce3f30b3a89bd7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_123f4206f30e0c7376b3076b177cb920
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 64, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e1ea61a59fa9688fe5b5cb9379fe9dc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2c584fd1118f519131b94737623b6208(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bbaf5d1c503713a54c84822ceedb3247(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1c98fa6ce6a211acb2959b697629eb57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_03af061a9a0c1e9312a124e93e94240e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f17c4a03dee78a4643daf1385d7a0a7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_daff1ac66075e434d8138f52f5c23901(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_575a9bba0ee5cd919aeb2fb114c9ea41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c0814ad203f975c141cf885e1439b97a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_819ed9e8ec4c73b7cfd0df6dbeeb90a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cd216b6735da078f61aaf912373c80a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_977b38f6025455b1bf16da8a1e8e4e3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a0dcb285ea4c5fa6dc04d7241bc83247(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0063b9a56eb0cb36e845b48650599f4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4d3b5720d3c38a288dccaacf17de2863(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ad2670672186f7f7086c4b5b6052129e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ea9127a86150fcbb6c13f7d72dfdae00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
            paddle.to_tensor([17], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7eded6e5ffc0d6ab7ee51899d2d34863(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([17], dtype='int64').reshape([1]),
            paddle.to_tensor([18], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_84bf02760cb726acd5259fb67ba7e120(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([18], dtype='int64').reshape([1]),
            paddle.to_tensor([19], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b016622e6973eb7666f536961f4918f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([19], dtype='int64').reshape([1]),
            paddle.to_tensor([20], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1e2f090825aeab17b6405d813ad1b915(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([20], dtype='int64').reshape([1]),
            paddle.to_tensor([21], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_00ce9fd57c97bd6e7a923ae885eab92f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([21], dtype='int64').reshape([1]),
            paddle.to_tensor([22], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a25ab537954acc4a01eff41b5f181de8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([22], dtype='int64').reshape([1]),
            paddle.to_tensor([23], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6ffb486ddc3ca561b1422762608c985b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([23], dtype='int64').reshape([1]),
            paddle.to_tensor([24], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e86fe294d6c9874bb8770f1d2e7ee63e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([24], dtype='int64').reshape([1]),
            paddle.to_tensor([25], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c39e286db3e42cd1705469a7a3d72d1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([25], dtype='int64').reshape([1]),
            paddle.to_tensor([26], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_95fb44aab2b53e6df5b08fe93cb48143(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([26], dtype='int64').reshape([1]),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c508a328c5757fc74aceedc4d6d27470(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
            paddle.to_tensor([28], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_103787398c4a7558a87ce672807dcb42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([28], dtype='int64').reshape([1]),
            paddle.to_tensor([29], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cbdc8575d5bc3553abe2f9eb5f999fe0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([29], dtype='int64').reshape([1]),
            paddle.to_tensor([30], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_48158079162dace7e98774ebee41eb62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([30], dtype='int64').reshape([1]),
            paddle.to_tensor([31], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a17740a5ff3b87226be4c77794112b51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([31], dtype='int64').reshape([1]),
            paddle.to_tensor([32], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e2a450fc4f755d9a7a4bafcc35f48d16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([32], dtype='int64').reshape([1]),
            paddle.to_tensor([33], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2db62be69e996b544ca66612c7d40765(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([33], dtype='int64').reshape([1]),
            paddle.to_tensor([34], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6af0a22016732455f684651abd5118cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([34], dtype='int64').reshape([1]),
            paddle.to_tensor([35], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1a627671f4e34d31906886b09634787d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([35], dtype='int64').reshape([1]),
            paddle.to_tensor([36], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d8791a28317eba4fc2aa281528734f1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([36], dtype='int64').reshape([1]),
            paddle.to_tensor([37], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b65b480ac539aa8af567faba578f88e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([37], dtype='int64').reshape([1]),
            paddle.to_tensor([38], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7bac06b7f2f62c5ffd6edcfef8c3b563(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([38], dtype='int64').reshape([1]),
            paddle.to_tensor([39], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_84c472cdff474e7bba18ef6b30ac7613(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([39], dtype='int64').reshape([1]),
            paddle.to_tensor([40], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8a55fca2b50061fc5fc53d5865ab702e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([40], dtype='int64').reshape([1]),
            paddle.to_tensor([41], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c5dbe7c698be799911b40f6a0b9d44ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([41], dtype='int64').reshape([1]),
            paddle.to_tensor([42], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4802d21ac3887521f42e604865620d3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([42], dtype='int64').reshape([1]),
            paddle.to_tensor([43], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ebdfa7b64f4e895ea4ffa1fdebe57292(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([43], dtype='int64').reshape([1]),
            paddle.to_tensor([44], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d58124e02d5418ec379e6c2f49b977ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([44], dtype='int64').reshape([1]),
            paddle.to_tensor([45], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3b9ef6c8cf72df8edf876f3e27919ea8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([45], dtype='int64').reshape([1]),
            paddle.to_tensor([46], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fb68ed30f1e00252c28db6bec49ef406(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([46], dtype='int64').reshape([1]),
            paddle.to_tensor([47], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_18c6e71ad31736019a8c44d863b0ab73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([47], dtype='int64').reshape([1]),
            paddle.to_tensor([48], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5f2e7f4b24772aa222d5a3d34352d2bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_253fad1030cb213cb1ea1d08de72a9df
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([48], dtype='int64').reshape([1]),
            paddle.to_tensor([49], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_fd2c04d2f4825e7ad9ea72ac7923493d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 11, 24, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_18835f2234b76db909a042691081d981(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd2c04d2f4825e7ad9ea72ac7923493d
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_24c348faa3150bfc29cfea793ad067e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd2c04d2f4825e7ad9ea72ac7923493d
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_859614572eb1d4eeaa9f578d68ed3c1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ea2b075f277df0f9769c469036c0260f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47b098d057c0de033368062ee6c186f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9480975e26ee7b6696ccdaef3f7f0235
    def get_inputs(self):
        return [
            paddle.to_tensor([2119, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9217b7beeafde9c391f8b3aef5f70c3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9480975e26ee7b6696ccdaef3f7f0235
    def get_inputs(self):
        return [
            paddle.to_tensor([2119, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_24022047511fdf364e3597cab7b65371(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 22, 6, 197, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d06d4ba24e776a22f0cf897e08dbf67b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24022047511fdf364e3597cab7b65371
    def get_inputs(self):
        return [
            paddle.uniform([2, 22, 6, 197, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_05d80624181b626a7d919a7fbfef2f98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24022047511fdf364e3597cab7b65371
    def get_inputs(self):
        return [
            paddle.uniform([2, 22, 6, 197, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d6b858e0bb4e2abcac74ce812280e850(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f861d96f50662e49b006201c2eb5d58a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6b858e0bb4e2abcac74ce812280e850
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_45dad4b3040c286b1a2eca5560fd49b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6b858e0bb4e2abcac74ce812280e850
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f769557366593d2f33e32458b8831d25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6b858e0bb4e2abcac74ce812280e850
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b534f392ce993c18d1d7604213d7ce4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6b858e0bb4e2abcac74ce812280e850
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f24ad6a147126de06789202c24394f2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6b858e0bb4e2abcac74ce812280e850
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9ebe5830755cf586d46515659482d73d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6b858e0bb4e2abcac74ce812280e850
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6b13d5c23d12ca36f063674c6a7e0b31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6b858e0bb4e2abcac74ce812280e850
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9f85c5159daaf6c4d125c2031e5ae152(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6b858e0bb4e2abcac74ce812280e850
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_beeff95986d350a6089448dc336bbc67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6b858e0bb4e2abcac74ce812280e850
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_60aa40cf17c8189529bd53e163e09e8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6b858e0bb4e2abcac74ce812280e850
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_39cd810553dd634d94ea312549ef5131(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6b858e0bb4e2abcac74ce812280e850
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e147b8b12124afafd5a0964ab5c3e4d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6b858e0bb4e2abcac74ce812280e850
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c77f1c40c294f2facfb34252aa28761b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6b858e0bb4e2abcac74ce812280e850
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f590b728e1f7edbe4b2105b7ce5e2cda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6b858e0bb4e2abcac74ce812280e850
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ed2026b3e8629f321de8b969a84e3304(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6b858e0bb4e2abcac74ce812280e850
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9123c6954f77efee914a08c04a5bee6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6b858e0bb4e2abcac74ce812280e850
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_9c2847df77f89a6a8960598823b2b88e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 1, 12, 1174, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8af7e163895200bb06f547d20580d047(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c2847df77f89a6a8960598823b2b88e
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 1174, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a5090f13173b02b19fa6e352722bbd19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c2847df77f89a6a8960598823b2b88e
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 1174, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c220bea8bda024646192ace8f5f7090d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c2847df77f89a6a8960598823b2b88e
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 1174, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d4a3952d540796912c52c02b679f8c83(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 86, 3, 198, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e8927d8893af8c349e82f01d5dc78898(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4a3952d540796912c52c02b679f8c83
    def get_inputs(self):
        return [
            paddle.uniform([3, 86, 3, 198, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c89316e35f409fd377674e1d4a698d3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4a3952d540796912c52c02b679f8c83
    def get_inputs(self):
        return [
            paddle.uniform([3, 86, 3, 198, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4c02e8dbb44b46c6e4190b67cd219335(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4a3952d540796912c52c02b679f8c83
    def get_inputs(self):
        return [
            paddle.uniform([3, 86, 3, 198, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cdee69ab42b9d16e1d0f94dcb1688c84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_14d53f860f9464d92eca41c6877197e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8727b67dd73bdef20851c6fefd11454b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c1b948f72b49798cbd5781ec6b0843b
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3aaec63e5cb39a51cd67f9d575513786(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c1b948f72b49798cbd5781ec6b0843b
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1b2dc6f1e3f8a1a8d27fe0eba724169a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2be559ce424338711c1cbec0c7d16111
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_32213ecef66619d3d4d981319a22bbea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2be559ce424338711c1cbec0c7d16111
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4ecb9138f0021b6edb980298b58d8397(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0238a9b7251f0d40866cdf80199c02f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4ecb9138f0021b6edb980298b58d8397(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0238a9b7251f0d40866cdf80199c02f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4ecb9138f0021b6edb980298b58d8397(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0238a9b7251f0d40866cdf80199c02f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d15db0d69c3343745a59359c3b2f9cd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2048, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b7775f69339549810d66f4612cc265df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2048, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_714dca9551891de5c71f29cdf77297ce(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 10, 8, 50, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7d8c019d8ac1e60ce6370b8ea911ce9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_714dca9551891de5c71f29cdf77297ce
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 8, 50, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a58f0e779fc35140165804b19987eeb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_714dca9551891de5c71f29cdf77297ce
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 8, 50, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_474c05484ed075e7d9de94ef30340a8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_714dca9551891de5c71f29cdf77297ce
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 8, 50, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_31c5f542173731900e3b7b45bd837f0f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 43, 1, 24, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_28e7fddca9e2692f0364ec9f42c34cef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31c5f542173731900e3b7b45bd837f0f
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7616f5efe2450bfa2a763177366ad4ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31c5f542173731900e3b7b45bd837f0f
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9849d307864fe9d8937e3747c743b3e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31c5f542173731900e3b7b45bd837f0f
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_1f6b5a569321115d4e4e31e9cdfc2a49(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 1, 6, 1025, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ccb560429acf87bffadf072eccc4a286(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f6b5a569321115d4e4e31e9cdfc2a49
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 6, 1025, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_96c983be6536244ed9b32e436d09591b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f6b5a569321115d4e4e31e9cdfc2a49
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 6, 1025, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f3cb2981eba90e02c7f76bc4fb452b81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f6b5a569321115d4e4e31e9cdfc2a49
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 6, 1025, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cdee69ab42b9d16e1d0f94dcb1688c84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_14d53f860f9464d92eca41c6877197e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cdee69ab42b9d16e1d0f94dcb1688c84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_14d53f860f9464d92eca41c6877197e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cdee69ab42b9d16e1d0f94dcb1688c84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_14d53f860f9464d92eca41c6877197e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4a294e62881976eb8e360468c5709a7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2048, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2b2c86e7ebca68c0c163aa4f5ceb88c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2048, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b6bc8475390dea836d5d07864c8db38d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9480975e26ee7b6696ccdaef3f7f0235
    def get_inputs(self):
        return [
            paddle.to_tensor([5606, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f2ec3ad594298e02ac7b7f1b922ed200(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9480975e26ee7b6696ccdaef3f7f0235
    def get_inputs(self):
        return [
            paddle.to_tensor([5606, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ddc5e38b09ab591fb6c9af49c7ffdbdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1b2bf71982cdba2694a56be835df15a
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6886b4652668c7f30e161fe3d529e7c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1b2bf71982cdba2694a56be835df15a
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c11e1dcbdf4c29e8baefd51f66c5e9be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1b2bf71982cdba2694a56be835df15a
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7dc5b880df3a50b020226673d300062f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff1b21f6c7c7c514facff0a4cd1b5e88
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 4, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b6160058404152c6c763aeb91212d85a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff1b21f6c7c7c514facff0a4cd1b5e88
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 4, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0520811acb435fb32f24030febc35672(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff1b21f6c7c7c514facff0a4cd1b5e88
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 4, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6da2dba6cf41c8144e6f4b99dccf3bf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9480975e26ee7b6696ccdaef3f7f0235
    def get_inputs(self):
        return [
            paddle.to_tensor([1036, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c9bd5e0a424087d690d9d97a7d86f383(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9480975e26ee7b6696ccdaef3f7f0235
    def get_inputs(self):
        return [
            paddle.to_tensor([1036, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_859614572eb1d4eeaa9f578d68ed3c1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ea2b075f277df0f9769c469036c0260f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e1daa0bc7b4a96205b4b16551de428ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d3785f190e39e98fa47ec1e321da0e4
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a66b7d09191c01b7055974b2e2bd56e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d3785f190e39e98fa47ec1e321da0e4
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_73f8a7d8da9284291ffc8e44776ecfa5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 97, 97], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f0afbf44686d598e536e887674d5541c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 97, 97], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_89285cdf4269e4d36a321f8163e8a2c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9480975e26ee7b6696ccdaef3f7f0235
    def get_inputs(self):
        return [
            paddle.to_tensor([1809, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4340936f89df4976db65ad7d7a696b79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9480975e26ee7b6696ccdaef3f7f0235
    def get_inputs(self):
        return [
            paddle.to_tensor([1809, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_eb13f6ae57d8e3b8b5ca78b703b1f815(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 86, 3, 197, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_db1b669bdfd8b0919c89b80c115870e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb13f6ae57d8e3b8b5ca78b703b1f815
    def get_inputs(self):
        return [
            paddle.uniform([3, 86, 3, 197, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_62074ec1a802eafaed035fe57770ca28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb13f6ae57d8e3b8b5ca78b703b1f815
    def get_inputs(self):
        return [
            paddle.uniform([3, 86, 3, 197, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1f12960a71a6be399c2ef6cdc9c0d6c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb13f6ae57d8e3b8b5ca78b703b1f815
    def get_inputs(self):
        return [
            paddle.uniform([3, 86, 3, 197, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ed6b73f16e18794bb6d6092a3df560f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d55aaaa33adefb17f34b84c755d178f9
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 4, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_729e8bdc01d10cec4681e78fe1b8eea8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d55aaaa33adefb17f34b84c755d178f9
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 4, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_38f0b858a1e75dae3ebdfea38bdd883f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d55aaaa33adefb17f34b84c755d178f9
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 4, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_af19925a98287eb51ff4c5fbc0232a5b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 10, 6, 197, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e037483bdf70570bd643f2c71a2ad569(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af19925a98287eb51ff4c5fbc0232a5b
    def get_inputs(self):
        return [
            paddle.uniform([2, 10, 6, 197, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1af51aa503b63ea0b949755c9c03f4ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af19925a98287eb51ff4c5fbc0232a5b
    def get_inputs(self):
        return [
            paddle.uniform([2, 10, 6, 197, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f6dcbd13e6ac359be84d414b6e28311f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 1, 12, 577, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0e6865c4953023961207887e31107164(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6dcbd13e6ac359be84d414b6e28311f
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 577, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1c212c6ce341812cd0989f41fc449762(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6dcbd13e6ac359be84d414b6e28311f
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 577, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_74f25d33b1b24a4ae09321abc33dcb26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6dcbd13e6ac359be84d414b6e28311f
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 577, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7dfdb5548be93c122f9202a91dc42b3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f063157bd604da0ae2ae05c617410ad
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 64, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c5b4868a4f5994a0f46251792619c74a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f063157bd604da0ae2ae05c617410ad
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 64, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0b19c74abc0fa4e9205f5ac551c62acd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f063157bd604da0ae2ae05c617410ad
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 64, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_a25656aa6082221b78367e1f8359591a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 1, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f3e84891dcbf5aaefd27e41a85a47c44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a25656aa6082221b78367e1f8359591a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[-0.4869559705257416, -0.008117705583572388, 0.1208193302154541, -0.040837496519088745]], [[-0.4454517662525177, 0.17571407556533813, 0.41662949323654175, -0.4325539469718933]], [[0.29227834939956665, -0.40047842264175415, 0.022228777408599854, -0.4592758119106293]], [[0.3593769669532776, -0.26735997200012207, 0.03356945514678955, 0.23702162504196167]], [[0.04704153537750244, 0.13100314140319824, -0.45369166135787964, 0.08746886253356934]], [[-0.30472972989082336, 0.0873795747756958, -0.13927185535430908, -0.2319345772266388]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9662b69f6e858cc0b90b7b380819a753(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a25656aa6082221b78367e1f8359591a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[-0.4869559705257416, -0.008117705583572388, 0.1208193302154541, -0.040837496519088745]], [[-0.4454517662525177, 0.17571407556533813, 0.41662949323654175, -0.4325539469718933]], [[0.29227834939956665, -0.40047842264175415, 0.022228777408599854, -0.4592758119106293]], [[0.3593769669532776, -0.26735997200012207, 0.03356945514678955, 0.23702162504196167]], [[0.04704153537750244, 0.13100314140319824, -0.45369166135787964, 0.08746886253356934]], [[-0.30472972989082336, 0.0873795747756958, -0.13927185535430908, -0.2319345772266388]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8d816cecad05c2eb3171afa551158fe1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a25656aa6082221b78367e1f8359591a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[-0.4869559705257416, -0.008117705583572388, 0.1208193302154541, -0.040837496519088745]], [[-0.4454517662525177, 0.17571407556533813, 0.41662949323654175, -0.4325539469718933]], [[0.29227834939956665, -0.40047842264175415, 0.022228777408599854, -0.4592758119106293]], [[0.3593769669532776, -0.26735997200012207, 0.03356945514678955, 0.23702162504196167]], [[0.04704153537750244, 0.13100314140319824, -0.45369166135787964, 0.08746886253356934]], [[-0.30472972989082336, 0.0873795747756958, -0.13927185535430908, -0.2319345772266388]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_71724779bff09b67dfef04bf54b4fe5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a25656aa6082221b78367e1f8359591a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[-0.4869559705257416, -0.008117705583572388, 0.1208193302154541, -0.040837496519088745]], [[-0.4454517662525177, 0.17571407556533813, 0.41662949323654175, -0.4325539469718933]], [[0.29227834939956665, -0.40047842264175415, 0.022228777408599854, -0.4592758119106293]], [[0.3593769669532776, -0.26735997200012207, 0.03356945514678955, 0.23702162504196167]], [[0.04704153537750244, 0.13100314140319824, -0.45369166135787964, 0.08746886253356934]], [[-0.30472972989082336, 0.0873795747756958, -0.13927185535430908, -0.2319345772266388]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_39c44bfb339def15ece36528aaf8437d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_123f4206f30e0c7376b3076b177cb920
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 64, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_798fc51913fb164464c70d7e966d5ff3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_123f4206f30e0c7376b3076b177cb920
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 64, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4f3e1dfd705b1190bce3f30b3a89bd7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_123f4206f30e0c7376b3076b177cb920
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 64, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_de6815d7089ec4a90cfaa323ece4233f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8fb1a0239b0d9299a9e0cc4a8c2e7cff
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f8abb13541a7b6fac023b092f86f9891(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8f87f669abcdb86c1b01768647cb8fe
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4fe913b85b951b8e10d0f16e58dc5198(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8f87f669abcdb86c1b01768647cb8fe
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_506460e6fe3520000415633e7e56b784(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9480975e26ee7b6696ccdaef3f7f0235
    def get_inputs(self):
        return [
            paddle.to_tensor([4179, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e3de3629903354d3239deb93cd932412(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9480975e26ee7b6696ccdaef3f7f0235
    def get_inputs(self):
        return [
            paddle.to_tensor([4179, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b466dc1071346547135dd43b4655f573(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b72c8a8cc4a0eff57d71cf709c61b656(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1169f9071ce5743e24091fa800b817
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_724ff24531c0eaec3753e32186df779f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 11, 12, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_44ddb2e186e49d3818cdd88b4231e124(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_724ff24531c0eaec3753e32186df779f
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fe3e80039185e48a72c91dfe834a8987(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_724ff24531c0eaec3753e32186df779f
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0f8786484e75f1745c7e58e81bb89f5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea5c52a37ea3ebdf749f0c7de615e713
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 320, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_383d16f35ce04a7a8f4da7a2d04bcfe5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea5c52a37ea3ebdf749f0c7de615e713
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 320, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e401816b5f61fefedddfbd553b921db3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea5c52a37ea3ebdf749f0c7de615e713
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 320, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a923655af77b36b441d0170cc91ef8fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9480975e26ee7b6696ccdaef3f7f0235
    def get_inputs(self):
        return [
            paddle.to_tensor([4662, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cd8bbffb3b3c58f9ab5fd6eb1c27727a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9480975e26ee7b6696ccdaef3f7f0235
    def get_inputs(self):
        return [
            paddle.to_tensor([4662, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_8c5599eeb84e7d1c890c0a7ea8664030(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 1, 8, 512, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bb41f9d78741b897c6808b35523f4aa3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c5599eeb84e7d1c890c0a7ea8664030
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 512, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4540054a95c82bbd89a9744c1b2aca93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c5599eeb84e7d1c890c0a7ea8664030
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 512, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_44ddb2e186e49d3818cdd88b4231e124(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_724ff24531c0eaec3753e32186df779f
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fe3e80039185e48a72c91dfe834a8987(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_724ff24531c0eaec3753e32186df779f
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_5e284aae983ede33fc33fe542408b868(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cbb35bd16dfa5d89e64989c269c98077(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e284aae983ede33fc33fe542408b868
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 7], dtype='int64').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f61f68d05cd9bce6c42aa7f909af9f9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e284aae983ede33fc33fe542408b868
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 7], dtype='int64').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_021f5cccc75390d0b35f7703e3d45955(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9480975e26ee7b6696ccdaef3f7f0235
    def get_inputs(self):
        return [
            paddle.to_tensor([3857, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fd2106db51c1bd4f5b69262cdbd19188(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9480975e26ee7b6696ccdaef3f7f0235
    def get_inputs(self):
        return [
            paddle.to_tensor([3857, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4502240c583fd622c7189a2bba811aa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b625734245499b5ce7e9f95a6ca197fa
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 100, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_98e7eafe0180b652b5ec4c0073192e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b625734245499b5ce7e9f95a6ca197fa
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 100, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0de193cee2611ab7e89571910bf91677(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b625734245499b5ce7e9f95a6ca197fa
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 100, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ca1e073ed5d4874d2a0c735d9fe97943(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 1, 6, 1174, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dafab470ba58ce4853b13cf9c86d9fc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca1e073ed5d4874d2a0c735d9fe97943
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 6, 1174, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9b1763b7972903bc93855173455c04d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca1e073ed5d4874d2a0c735d9fe97943
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 6, 1174, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f1855f64d2dc65c84ba2e64b87307432(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca1e073ed5d4874d2a0c735d9fe97943
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 6, 1174, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_18835f2234b76db909a042691081d981(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd2c04d2f4825e7ad9ea72ac7923493d
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_24c348faa3150bfc29cfea793ad067e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd2c04d2f4825e7ad9ea72ac7923493d
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_0aacfe9c3b4c283dd9c968065a11802c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 1960, 4, 16, 6], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8b061b8e6c12ece156727d2f6afa6586(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0aacfe9c3b4c283dd9c968065a11802c
    def get_inputs(self):
        return [
            paddle.uniform([2, 1960, 4, 16, 6], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_450c1003e75557df729ea5b575844789(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0aacfe9c3b4c283dd9c968065a11802c
    def get_inputs(self):
        return [
            paddle.uniform([2, 1960, 4, 16, 6], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_32d7c083c791b8d5ddbbb385276e7151(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a25656aa6082221b78367e1f8359591a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.2764121890068054, -0.3669598698616028, -0.2626584768295288, 0.13232702016830444]], [[0.44474995136260986, 0.2665559649467468, 0.3600660562515259, -0.12660950422286987]], [[0.41293424367904663, -0.499341756105423, -0.2814410924911499, -0.04606747627258301]], [[0.10607725381851196, 0.22331172227859497, -0.048095256090164185, 0.02749711275100708]], [[-0.18778502941131592, -0.02420675754547119, 0.13080501556396484, 0.06155425310134888]], [[0.20700228214263916, -0.06112822890281677, -0.43966054916381836, -0.05186641216278076]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_de2eb3d818379f86ba91f19634d2d14f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a25656aa6082221b78367e1f8359591a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.2764121890068054, -0.3669598698616028, -0.2626584768295288, 0.13232702016830444]], [[0.44474995136260986, 0.2665559649467468, 0.3600660562515259, -0.12660950422286987]], [[0.41293424367904663, -0.499341756105423, -0.2814410924911499, -0.04606747627258301]], [[0.10607725381851196, 0.22331172227859497, -0.048095256090164185, 0.02749711275100708]], [[-0.18778502941131592, -0.02420675754547119, 0.13080501556396484, 0.06155425310134888]], [[0.20700228214263916, -0.06112822890281677, -0.43966054916381836, -0.05186641216278076]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_51f75cf4c388e0b00f9a7b185a5075bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a25656aa6082221b78367e1f8359591a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.2764121890068054, -0.3669598698616028, -0.2626584768295288, 0.13232702016830444]], [[0.44474995136260986, 0.2665559649467468, 0.3600660562515259, -0.12660950422286987]], [[0.41293424367904663, -0.499341756105423, -0.2814410924911499, -0.04606747627258301]], [[0.10607725381851196, 0.22331172227859497, -0.048095256090164185, 0.02749711275100708]], [[-0.18778502941131592, -0.02420675754547119, 0.13080501556396484, 0.06155425310134888]], [[0.20700228214263916, -0.06112822890281677, -0.43966054916381836, -0.05186641216278076]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be3a9fca9221194e37054dac03b3225(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a25656aa6082221b78367e1f8359591a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.2764121890068054, -0.3669598698616028, -0.2626584768295288, 0.13232702016830444]], [[0.44474995136260986, 0.2665559649467468, 0.3600660562515259, -0.12660950422286987]], [[0.41293424367904663, -0.499341756105423, -0.2814410924911499, -0.04606747627258301]], [[0.10607725381851196, 0.22331172227859497, -0.048095256090164185, 0.02749711275100708]], [[-0.18778502941131592, -0.02420675754547119, 0.13080501556396484, 0.06155425310134888]], [[0.20700228214263916, -0.06112822890281677, -0.43966054916381836, -0.05186641216278076]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_28e7fddca9e2692f0364ec9f42c34cef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31c5f542173731900e3b7b45bd837f0f
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7616f5efe2450bfa2a763177366ad4ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31c5f542173731900e3b7b45bd837f0f
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9849d307864fe9d8937e3747c743b3e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31c5f542173731900e3b7b45bd837f0f
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()