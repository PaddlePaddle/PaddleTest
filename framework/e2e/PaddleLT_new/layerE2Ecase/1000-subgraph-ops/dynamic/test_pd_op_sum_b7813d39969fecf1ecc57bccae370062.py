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



class PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f855bfc2b279955ae5f49f3c25942cc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5a42f8c5ef2aee1ea59eab279e5c1918(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 136, 136], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_2964f274495c2aa234a5be8c37975879(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_95984dec11fb700a9bcadecfd789e69b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2964f274495c2aa234a5be8c37975879
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_95984dec11fb700a9bcadecfd789e69b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2964f274495c2aa234a5be8c37975879
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_83af1443bc4a353dad270eb5eb36383f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2964f274495c2aa234a5be8c37975879
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.48773711919784546, 0.6488420963287354]], [[0.13296863436698914, 0.7793918251991272]], [[0.6046473979949951, 0.1745527684688568]], [[0.05936390906572342, 0.28628185391426086]], [[0.03372789919376373, 0.20185533165931702]], [[0.024453697726130486, 0.15137672424316406]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_789a41968acd9badc40fc7c3a246ef02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2964f274495c2aa234a5be8c37975879
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.009278914891183376, 0.006461446173489094]], [[0.715639591217041, 0.8949653506278992]], [[0.2983335554599762, 0.20823298394680023]], [[0.01422260981053114, 0.38681530952453613]], [[0.07722488790750504, 0.00014375177852343768]], [[0.26053765416145325, 0.32827335596084595]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_884a142590e018058dfbd87120671753(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a85f128e03fdb1f07a20e6c01e9cc479(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a85f128e03fdb1f07a20e6c01e9cc479(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ccfcd809da40afe8153c710632c845f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 100], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ced0d018def9586f4026d70772d35867(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f41137c8bc0e6e804ff33414d1599a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ced0d018def9586f4026d70772d35867
    def get_inputs(self):
        return [
            paddle.uniform([100, 2, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_13d8cbc543de467b7a700dc89aab3e2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_13d8cbc543de467b7a700dc89aab3e2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_465c80745b27c5a7564c0c86864e28cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1025, 1025], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e74dda2b8bff943aaaebccd0d76ec54d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ced0d018def9586f4026d70772d35867
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6080f86bc03c604d5d5309d73e5991e3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ecbe271f584bf079ec6537478250709c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6080f86bc03c604d5d5309d73e5991e3
    def get_inputs(self):
        return [
            paddle.uniform([1524, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_69ebab389651bb9406666743831d3ce9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6080f86bc03c604d5d5309d73e5991e3
    def get_inputs(self):
        return [
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_12d561f68c831a12544831886a48e5bc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_81dd3a846cfab13eebfc808564f6c5ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12d561f68c831a12544831886a48e5bc
    def get_inputs(self):
        return [
            paddle.uniform([1524, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_69ebab389651bb9406666743831d3ce9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6080f86bc03c604d5d5309d73e5991e3
    def get_inputs(self):
        return [
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_f36da0438e0a51874dd1cdb6fc23ee7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ced0d018def9586f4026d70772d35867
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_136398d425fabe6438524de7937f5740(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6080f86bc03c604d5d5309d73e5991e3
    def get_inputs(self):
        return [
            paddle.uniform([2340, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_71d540c28bf065167ca807f92d76003b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6080f86bc03c604d5d5309d73e5991e3
    def get_inputs(self):
        return [
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_daf3ce5c21ae84d6ca26252d5953643e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12d561f68c831a12544831886a48e5bc
    def get_inputs(self):
        return [
            paddle.uniform([2340, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_71d540c28bf065167ca807f92d76003b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6080f86bc03c604d5d5309d73e5991e3
    def get_inputs(self):
        return [
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_afc91ceeddac1a73d874a5c09bf77b10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_31217320838d295d6055009678f636e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_31217320838d295d6055009678f636e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8f987c7ffec509e063f5894d5a90280e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a361704f4fc396703403d4a7f86f764b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([6, 2304, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a361704f4fc396703403d4a7f86f764b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([6, 2304, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_80c6b224ada72bb91462d6166c70bb01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_80c6b224ada72bb91462d6166c70bb01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dbc07d6f105a0bdd9a2ee0d887c20afb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 120, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_70a35e2a1dcfa62419c784df45e558c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_70a35e2a1dcfa62419c784df45e558c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_309748020e97834feafaf9bfe200d150(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_309748020e97834feafaf9bfe200d150(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_90567394057c604ffda76eb9c9b6b511(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_90567394057c604ffda76eb9c9b6b511(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_beb97d19f8538ba6bf4ad3cca9f6dad4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_beb97d19f8538ba6bf4ad3cca9f6dad4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cdb51feeac32eb405c76c893da92cc3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cdb51feeac32eb405c76c893da92cc3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a3f7c07b94b075ff13b9c492ff3345e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a3f7c07b94b075ff13b9c492ff3345e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3a9038e5d3d3e1132f9583a107005a4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2a5f9ed5c4fcbeb2f4e31fb83db5fd09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([54, 197, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2a5f9ed5c4fcbeb2f4e31fb83db5fd09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([54, 197, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2c53e21d47a23f66b8eefbdf56afcd92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 197, 197], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1f98cd06e79b31aab254343fe4fad531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1f98cd06e79b31aab254343fe4fad531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d1646cb24811512dd7c2ba042e6b62e8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4d2c20d7e49f46d7a45a3e3b982a6dce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1646cb24811512dd7c2ba042e6b62e8
    def get_inputs(self):
        return [
            paddle.uniform([8816], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_99ae62d2b884bc8aea5dcf61129f352c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_0423c0e2c3b01ba49117fd6321aa4d76(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_47b4edc687c14828c29880c4a6392c5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0423c0e2c3b01ba49117fd6321aa4d76
    def get_inputs(self):
        return [
            paddle.to_tensor([1.5373048782348633, 1.1550887823104858, 2.89843487739563, 1.255260944366455, 2.7917590141296387, 1.4412775039672852], dtype='float32').reshape([6]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_94de098bf59f51bc5f318e663b40eef7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8b033e3957c4dfb75e5749e54d6c566f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94de098bf59f51bc5f318e663b40eef7
    def get_inputs(self):
        return [
            paddle.uniform([22, 4, 16, 49, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e862be66d62d6c80e0e548521304bea7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6080f86bc03c604d5d5309d73e5991e3
    def get_inputs(self):
        return [
            paddle.uniform([47, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_02bed273bb67bfb32905a315870f310a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d64af61e10fbe3ca607cb0c81a019169(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94de098bf59f51bc5f318e663b40eef7
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1e7a5b3347ee366e721c8851305f1f28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94de098bf59f51bc5f318e663b40eef7
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_707d73fa0e297c771a933f938bf6b755(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1646cb24811512dd7c2ba042e6b62e8
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_0e2fa19ae58bc24cf729976bd53c8802(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 168, 168], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_cff5e47bad306e1492f516529a701448(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cff5e47bad306e1492f516529a701448(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8435aafbb1f070d58bda3cca5546c92b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8435aafbb1f070d58bda3cca5546c92b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f6c55bf64bebf6840b097310e9695f75(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, paddle.float32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f301326633b026c25866ec8e4a167ad8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6c55bf64bebf6840b097310e9695f75
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_113c4bec7e206cc794e5d83f3a801c34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6c55bf64bebf6840b097310e9695f75
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4, 50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d79cd8374691e4580640898179e9e934(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d79cd8374691e4580640898179e9e934(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_78b02e2932137ef331529ccb0a6ead88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 640, 640], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c2e2c28f929cdf46c3cfb6c20b29a0e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94de098bf59f51bc5f318e663b40eef7
    def get_inputs(self):
        return [
            paddle.uniform([22, 2, 16, 9, 112, 112], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c4a60888a77353c06b26cccd05ef5db5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 104, 104], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_cff5e47bad306e1492f516529a701448(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cff5e47bad306e1492f516529a701448(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d54fc3eccabfe6466062962d5987349f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 13, 13], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9c60b9fdbf891326071bda09f247c834(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ced0d018def9586f4026d70772d35867
    def get_inputs(self):
        return [
            paddle.uniform([300, 2, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a7b2a625675cff5970afc2f4947e9f7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 52, 52], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1940dffc139eb8126376e3e0640d4fed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6c55bf64bebf6840b097310e9695f75
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8afbbb2ff63d91bd9f3d51e0274f698d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6c55bf64bebf6840b097310e9695f75
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4, 100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7249a15c7c4d8945708dd7d316566f91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([1, 21, 16384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_80c6b224ada72bb91462d6166c70bb01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_80c6b224ada72bb91462d6166c70bb01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f9166a921886647a1997a421d9130cde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f9166a921886647a1997a421d9130cde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8260ef6ad34a1762ce8f994a51a84de5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 200, 200], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_84e9c22e6f53248136bc26ce81b95b01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_84e9c22e6f53248136bc26ce81b95b01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_66e210b623775461e400a84b6d411ba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_37fe9c1f7499bd15583b673b90b7a473(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3c471e402869057475e35147e81ee2dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3c471e402869057475e35147e81ee2dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2a5c0da63926010cef345fa380e45cec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f99826b052c937dbdb48876e7a95458a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 40, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_fae5ee739975056db6210e001c2c8317(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1646cb24811512dd7c2ba042e6b62e8
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.288908988237381, -0.018105916678905487, -0.19367942214012146, -0.11516457796096802], dtype='float32').reshape([4]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_3508a4e7b93a1637044a159fe39e61a9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4bfc7749aeb24364c1f90430b6fc1584(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3508a4e7b93a1637044a159fe39e61a9
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_2b320f6263085ddb5d72a73bb136b086(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2b320f6263085ddb5d72a73bb136b086(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7a6f84676bba664e8dee136c0b8e514d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6c55bf64bebf6840b097310e9695f75
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 7, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_44c1178ed684866a82fb841189718dec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6c55bf64bebf6840b097310e9695f75
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4, 7, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6ccc0e17355a206b3c4060ea7d86b4c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_680c328eda03102b84f4a373b26ef538(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2d646d47a0edbf09821b0c52a7084278(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([1, 60800, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2d646d47a0edbf09821b0c52a7084278(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([1, 60800, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c42fec69c30bd30a07cf624d072e7f73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c42fec69c30bd30a07cf624d072e7f73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_70a35e2a1dcfa62419c784df45e558c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_70a35e2a1dcfa62419c784df45e558c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_91b681264a1dc396b8a85f9f7c6d86fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd1fd8aca73a476779e8ace375c8e2a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd1fd8aca73a476779e8ace375c8e2a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_94e53637d00710ef829cbf4d8b0bbdc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 1, 91], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_370f5e8bfb78edec32b40cecb967f58c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_370f5e8bfb78edec32b40cecb967f58c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6c3f9dfe2f5f856982c110cc222610b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6c3f9dfe2f5f856982c110cc222610b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f1f750e4a79967692533e7e6b477ecf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 21, 21], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5e22bde2ac5b6caba20c78fd33eda454(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 46, 46], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f0d00fd32a064172bd544557be60ad9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1646cb24811512dd7c2ba042e6b62e8
    def get_inputs(self):
        return [
            paddle.uniform([551], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_3c471e402869057475e35147e81ee2dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3c471e402869057475e35147e81ee2dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8435aafbb1f070d58bda3cca5546c92b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8435aafbb1f070d58bda3cca5546c92b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7a5c8886c17c6c0b0a6eff351314973c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 256, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_70a35e2a1dcfa62419c784df45e558c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_70a35e2a1dcfa62419c784df45e558c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0c45cc110d22e3ecea39346985eeac4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 80, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8435aafbb1f070d58bda3cca5546c92b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8435aafbb1f070d58bda3cca5546c92b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_efd8318899ea6844c69216089f1f6d8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 13, 13], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6248e3b107360867960850fc8ea0040f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6248e3b107360867960850fc8ea0040f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0fa108988ada96be49459efa0610aa03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 160, 160], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8dbe00bc75971d95e7b19510189947ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4de41814bcb246c9a513b51ecc3be1f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4de41814bcb246c9a513b51ecc3be1f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4de41814bcb246c9a513b51ecc3be1f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4de41814bcb246c9a513b51ecc3be1f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_56d24dcff096bbf29d99d7dca22f5b59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ced0d018def9586f4026d70772d35867
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fa858c99137d37c6f3575ca24457a54f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6080f86bc03c604d5d5309d73e5991e3
    def get_inputs(self):
        return [
            paddle.uniform([2047, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_c11819dd0f0e7dcfb73448cf269d75f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6080f86bc03c604d5d5309d73e5991e3
    def get_inputs(self):
        return [
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_797d9bab51b682d36b0ff5cedf47b27a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12d561f68c831a12544831886a48e5bc
    def get_inputs(self):
        return [
            paddle.uniform([2047, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c11819dd0f0e7dcfb73448cf269d75f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6080f86bc03c604d5d5309d73e5991e3
    def get_inputs(self):
        return [
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_956a27a5f91cf56aee5f2760854c8fe0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 23, 23], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5878fa2bd1de78aefbe32231703e191f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1646cb24811512dd7c2ba042e6b62e8
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_f291a394c2a73060742b8e823d4fc474(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6080f86bc03c604d5d5309d73e5991e3
    def get_inputs(self):
        return [
            paddle.uniform([103, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_48e3bab564fb98d8f064d7fc32fdce43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_48e3bab564fb98d8f064d7fc32fdce43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bf65aec8b3689b3d743361ea82d5074f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bf65aec8b3689b3d743361ea82d5074f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_aa7ce8b80f5497e4fa63ba64edcea25e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 320], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3084e99aaa3ff2ebc93d4706f461cdd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 96, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d7dc0b32850c19e358cb71bc04583690(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94de098bf59f51bc5f318e663b40eef7
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_65e975fd7da26d08838b42b2364b5bcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 112, 112], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_fde77faf05f1ed4bb80f2983fb61cc07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1646cb24811512dd7c2ba042e6b62e8
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0354757122695446, -0.19871774315834045, 0.08630979806184769, -0.04619042947888374, -0.2942329943180084, 0.23153549432754517, -0.018004804849624634, -0.20709450542926788, 0.29523739218711853, 0.11242446303367615, 0.27399760484695435, -0.2284012883901596, -0.2642057240009308, -0.14949648082256317, -0.16313910484313965, 0.0778934434056282, 0.15521733462810516, -0.0375627726316452, 0.24336369335651398, 0.12175900489091873], dtype='float32').reshape([20]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_f7a793de34efac4cc0f037bbf0b17b1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f7a793de34efac4cc0f037bbf0b17b1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d3d009964d4bb775127629f32acc2913(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ced0d018def9586f4026d70772d35867
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ab46bde0969d65fc513eef6f353eb5a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6080f86bc03c604d5d5309d73e5991e3
    def get_inputs(self):
        return [
            paddle.uniform([1813, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_6883aec64ab93eb08bf9ee8b3f78697f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6080f86bc03c604d5d5309d73e5991e3
    def get_inputs(self):
        return [
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_514648c95ccd6b0443ca1a7e48bda5fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12d561f68c831a12544831886a48e5bc
    def get_inputs(self):
        return [
            paddle.uniform([1813, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6883aec64ab93eb08bf9ee8b3f78697f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6080f86bc03c604d5d5309d73e5991e3
    def get_inputs(self):
        return [
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_1dd7e83623a98288dd1b2aa27fd6b4b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 34, 34], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1f804068a640c94cbea93aad77a377e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94de098bf59f51bc5f318e663b40eef7
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_57c2f2a55901bd72fec293b57594c4f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_57c2f2a55901bd72fec293b57594c4f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4e5286c8bfa281b89591ea1c17bfedf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_699eb4e5eb57d0f1079fefbd78a0e769(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_35791f47132cf06a81abfcf04ead0955(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699eb4e5eb57d0f1079fefbd78a0e769
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 34], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d17abf478ac5f1512be3ce5a1bdda1d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1646cb24811512dd7c2ba042e6b62e8
    def get_inputs(self):
        return [
            paddle.to_tensor([0.023478882387280464, -0.07743896543979645, -0.29410916566848755, -0.16785748302936554, -0.16195517778396606, -0.25259315967559814, 0.18069225549697876, 0.27407142519950867, -0.08581829816102982, -0.08196482807397842, -0.12156025320291519, -0.08806869387626648, 0.10745064169168472, 0.041386689990758896, -0.16404861211776733, 0.2426360696554184], dtype='float32').reshape([16]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_fd5cf4623d0b17c7a50f8758eb5135fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1646cb24811512dd7c2ba042e6b62e8
    def get_inputs(self):
        return [
            paddle.uniform([4875], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_bb22e6cb81610825b82e201bcbcffd6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([4, 2304, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bb22e6cb81610825b82e201bcbcffd6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([4, 2304, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd2c9b2c9927ef0426be968f2b07ad36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([1, 21760, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd2c9b2c9927ef0426be968f2b07ad36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([1, 21760, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bd342875ced6544622774e53670501bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94de098bf59f51bc5f318e663b40eef7
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04a8b97edf27ed602203b33d76cad42a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 256, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_382c6e275a1c1b6a58af992ffb45ba51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94de098bf59f51bc5f318e663b40eef7
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_172cdde2963d8eed64b906802ffe968d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([54, 198, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_172cdde2963d8eed64b906802ffe968d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([54, 198, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6d9dcf0b98fca0f0eead10a8f86eba58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 198, 198], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0f5e889824ac41512573cdc08ba25c2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 19, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_532f9f7040135376d7c1df2eeecfa9a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ced0d018def9586f4026d70772d35867
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a0cebe38f82ae66fdf0834bfa1d3b1d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6080f86bc03c604d5d5309d73e5991e3
    def get_inputs(self):
        return [
            paddle.uniform([3061, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_61ccfc5985c0d3d334643d683c2cfeef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6080f86bc03c604d5d5309d73e5991e3
    def get_inputs(self):
        return [
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_8b3f739b78578016e09651f51fd8d115(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12d561f68c831a12544831886a48e5bc
    def get_inputs(self):
        return [
            paddle.uniform([3061, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_61ccfc5985c0d3d334643d683c2cfeef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6080f86bc03c604d5d5309d73e5991e3
    def get_inputs(self):
        return [
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_dd9a4cceba6e01d53178b8097967a8e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_49f0ad3bcb964809f765697e98d21cba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cf66e5e39b33847fcb771dcce4a89ff3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cf66e5e39b33847fcb771dcce4a89ff3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8f987c7ffec509e063f5894d5a90280e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd1fd8aca73a476779e8ace375c8e2a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd1fd8aca73a476779e8ace375c8e2a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e29ff33b8882281e4bf2f0cc5eba92fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ee6d85c29123ba4fe2b8eab61220610f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 52, 52], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_84e9c22e6f53248136bc26ce81b95b01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_84e9c22e6f53248136bc26ce81b95b01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bd342875ced6544622774e53670501bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94de098bf59f51bc5f318e663b40eef7
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dc59aacd8b2cfbfc43342fd9dcaf4cb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 68, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a85f128e03fdb1f07a20e6c01e9cc479(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a85f128e03fdb1f07a20e6c01e9cc479(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8216cb06016f27a718c9d0c69dff1735(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 32768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6c3f9dfe2f5f856982c110cc222610b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6c3f9dfe2f5f856982c110cc222610b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b2d544ebe37ac250301b4ac416fcabe5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 60, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5c1f442c201333b84e9d28d98e49937c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5c1f442c201333b84e9d28d98e49937c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1e7a5b3347ee366e721c8851305f1f28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94de098bf59f51bc5f318e663b40eef7
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3c2fe7703fa25deb229609898f7ff85d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3c2fe7703fa25deb229609898f7ff85d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_aa9430a5151928bd39636f2c1d7cb625(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1f98cd06e79b31aab254343fe4fad531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1f98cd06e79b31aab254343fe4fad531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c34e4940d78d975e50ab05f9a0072835(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ced0d018def9586f4026d70772d35867
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0351114bc6654681db4b95005d6bd6d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6080f86bc03c604d5d5309d73e5991e3
    def get_inputs(self):
        return [
            paddle.uniform([2062, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_a120d65ac349e8ff601245baa9ef80b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6080f86bc03c604d5d5309d73e5991e3
    def get_inputs(self):
        return [
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_d580f26cea35e5bc98bbd57685af235e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12d561f68c831a12544831886a48e5bc
    def get_inputs(self):
        return [
            paddle.uniform([2062, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a120d65ac349e8ff601245baa9ef80b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6080f86bc03c604d5d5309d73e5991e3
    def get_inputs(self):
        return [
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_e63779c26e6def6260be1e254a606e39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 256, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e967632428e41bb0c2b4be4031bf3dd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 52, 52], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a3f7c07b94b075ff13b9c492ff3345e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a3f7c07b94b075ff13b9c492ff3345e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_af0cbb86e22ac8e8dcf1f8997220728e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_af0cbb86e22ac8e8dcf1f8997220728e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bba850d9d17cf6736fa11663a5df9cae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1174, 1174], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c938d66101465984329bc39e5ea1bdb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_382c6e275a1c1b6a58af992ffb45ba51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94de098bf59f51bc5f318e663b40eef7
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f2758b6d6e6e50ab13ecab0887a6bff7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1646cb24811512dd7c2ba042e6b62e8
    def get_inputs(self):
        return [
            paddle.uniform([950], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_577b498868d950cfd9ad22a407171785(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_dabd95c62078ae79d56003418bc48757(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_64acd96040687358213b5ca8f14982dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([86, 198, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_64acd96040687358213b5ca8f14982dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([86, 198, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_627ce72208713e6af31880f914f0dd72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 198, 198], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7b8856d03e064d4650a9ae83f34b17c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_35a7614aee28d8b364d4f5bae2f15e23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 15, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_84a7477af5f2c3fd236173f6c166a6f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8435aafbb1f070d58bda3cca5546c92b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8435aafbb1f070d58bda3cca5546c92b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9fb20c8e5200e36f506b6000a9d5ccb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1646cb24811512dd7c2ba042e6b62e8
    def get_inputs(self):
        return [
            paddle.uniform([247], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_ea2f958ab23678d3e86e78018f917491(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ea2f958ab23678d3e86e78018f917491(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c4ee6ad8c3f0c81f2e19e53337198e55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 30, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_01713229c51dac4a917a72978b64ab9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6d0851498d16eb79dbd346307db36994(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8a79f2f5d3a12b3f455fcdab8a8cd470(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 8192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c42fec69c30bd30a07cf624d072e7f73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c42fec69c30bd30a07cf624d072e7f73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_370f5e8bfb78edec32b40cecb967f58c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_370f5e8bfb78edec32b40cecb967f58c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3c2fe7703fa25deb229609898f7ff85d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3c2fe7703fa25deb229609898f7ff85d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2751e96b024f8dd1a6a6d0f7fba086c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 50, 50], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f149e5ddf71dfca9e06252634fc34ff3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6080f86bc03c604d5d5309d73e5991e3
    def get_inputs(self):
        return [
            paddle.uniform([56, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_a3f7c07b94b075ff13b9c492ff3345e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a3f7c07b94b075ff13b9c492ff3345e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_63e8c29362b6506a3de9de02272b38cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 160, 160], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3a5052dae8c003d7ff28e375d493ea15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([6, 576, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3a5052dae8c003d7ff28e375d493ea15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([6, 576, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6a32bc515fab2f3da8d8c520bd73f5d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6a32bc515fab2f3da8d8c520bd73f5d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cd3d7fc1c7f80779db1febb16af8f134(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1025, 1025], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_02abc8b56ffd7a251f33ab5bd2d3d099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_02abc8b56ffd7a251f33ab5bd2d3d099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_eb0f3e5db512755edd64f63275d15c80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_eb0f3e5db512755edd64f63275d15c80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_995589628506760554ef1dd5ed5c37b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_995589628506760554ef1dd5ed5c37b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1feea20828f14924201d6075a338f316(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1feea20828f14924201d6075a338f316(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d686e3ff77a6d5a9be4549f30ae78573(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e37be567d274564dbe4d15d6bb8790b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 84, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_36ed97416640ac92fa10715135be959f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1646cb24811512dd7c2ba042e6b62e8
    def get_inputs(self):
        return [
            paddle.uniform([70], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_ba9bf08dcc3c2c2f248c6d5b0227e4ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_648c74e7a7845b6d2f3ca45f6dad945b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ced0d018def9586f4026d70772d35867
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b249c59b4cfb3b41a70357a82538011d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6080f86bc03c604d5d5309d73e5991e3
    def get_inputs(self):
        return [
            paddle.uniform([5526, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_c38d8f411c8ba4c341caac7a0c993749(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6080f86bc03c604d5d5309d73e5991e3
    def get_inputs(self):
        return [
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_2bba4a0f1c887dd22fc57d29e07ee104(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12d561f68c831a12544831886a48e5bc
    def get_inputs(self):
        return [
            paddle.uniform([5526, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c38d8f411c8ba4c341caac7a0c993749(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6080f86bc03c604d5d5309d73e5991e3
    def get_inputs(self):
        return [
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_68802079f247282837fbe68f1c2112b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_63a20a02b48dabcf0ecbc0cf622f82cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 42, 42], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6151e27fc340ad0d4bb2b583a00d9d3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d91212e7387080b66950f5dfb9dcb578(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ced0d018def9586f4026d70772d35867
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8222667895b42943173286e513b1da82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6080f86bc03c604d5d5309d73e5991e3
    def get_inputs(self):
        return [
            paddle.uniform([1071, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_68c2eec32eafb495eebf94c9ae8caffd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6080f86bc03c604d5d5309d73e5991e3
    def get_inputs(self):
        return [
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_04951098696a95b95b77f306c294fe1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12d561f68c831a12544831886a48e5bc
    def get_inputs(self):
        return [
            paddle.uniform([1071, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_68c2eec32eafb495eebf94c9ae8caffd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6080f86bc03c604d5d5309d73e5991e3
    def get_inputs(self):
        return [
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_3a60fc2115a2375920958cef579476ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6080f86bc03c604d5d5309d73e5991e3
    def get_inputs(self):
        return [
            paddle.uniform([53, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_5c1f442c201333b84e9d28d98e49937c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5c1f442c201333b84e9d28d98e49937c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d3d009964d4bb775127629f32acc2913(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ced0d018def9586f4026d70772d35867
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_60634d4f8e777c99d5c54cf092036354(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6080f86bc03c604d5d5309d73e5991e3
    def get_inputs(self):
        return [
            paddle.uniform([1760, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_37f963f1a7e80f7f3dda7fef939362be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6080f86bc03c604d5d5309d73e5991e3
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_2ec53df3b9cd115d8e1e19c2e9b1b898(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12d561f68c831a12544831886a48e5bc
    def get_inputs(self):
        return [
            paddle.uniform([1760, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_37f963f1a7e80f7f3dda7fef939362be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6080f86bc03c604d5d5309d73e5991e3
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_47c6cd8170cfe5a248463f07613027b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([86, 197, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47c6cd8170cfe5a248463f07613027b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([86, 197, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ac5ae70033f669ed2966a1618a49528c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 197, 197], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5878fa2bd1de78aefbe32231703e191f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1646cb24811512dd7c2ba042e6b62e8
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_e6635ffa6e927ba2e31308ece8c58837(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6080f86bc03c604d5d5309d73e5991e3
    def get_inputs(self):
        return [
            paddle.uniform([84, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_9e52f580ae2027f8bed1047fc03a8c95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6c55bf64bebf6840b097310e9695f75
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 13, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ad1e017eba12d6d1fdef83af34fd1751(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6c55bf64bebf6840b097310e9695f75
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4, 13, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6ea450627d1a9b0954386c5e07f8924c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 17, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_58dced482ade70e9cc101a78aed32dbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 13, 13], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ce9de96b47a08b52beddcd2c87252537(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1646cb24811512dd7c2ba042e6b62e8
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.23329107463359833, -0.20089885592460632, -0.1679246425628662, 0.10302043706178665, 0.14582417905330658, -0.054320961236953735, -0.2298935353755951, -0.15897536277770996, 0.07082629203796387, -0.0012911971425637603, 0.04014170914888382, 0.005387966986745596, 0.07408301532268524, -0.22553326189517975, -0.17665284872055054, 0.12102960050106049, 0.21869981288909912, -0.11437982320785522, 0.24512448906898499, 0.08682923763990402, 0.04303419962525368, 0.23399941623210907, -0.15843921899795532, 0.07024722546339035], dtype='float32').reshape([24]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_1f804068a640c94cbea93aad77a377e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94de098bf59f51bc5f318e663b40eef7
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d465a862f7c7dabf8c9b30783aa3d9e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 4096], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_82ddedfeb86e99d65d83059ddc5f7c7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1646cb24811512dd7c2ba042e6b62e8
    def get_inputs(self):
        return [
            paddle.uniform([17571], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_4a3199401be7103404443e5df4824c1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94de098bf59f51bc5f318e663b40eef7
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 16, 9, 112, 112], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_309748020e97834feafaf9bfe200d150(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_309748020e97834feafaf9bfe200d150(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_90567394057c604ffda76eb9c9b6b511(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_90567394057c604ffda76eb9c9b6b511(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_beb97d19f8538ba6bf4ad3cca9f6dad4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_beb97d19f8538ba6bf4ad3cca9f6dad4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cdb51feeac32eb405c76c893da92cc3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cdb51feeac32eb405c76c893da92cc3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c6a2d6ad9daab1f87b396fd1b6e2cbd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c6a2d6ad9daab1f87b396fd1b6e2cbd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_de9cc5168d99e7cd2ff72ada29258bd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 577, 577], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d64af61e10fbe3ca607cb0c81a019169(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94de098bf59f51bc5f318e663b40eef7
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f143ab8faa6fbd58c4e0ee6f1d0255bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1646cb24811512dd7c2ba042e6b62e8
    def get_inputs(self):
        return [
            paddle.uniform([2204], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_a3f7c07b94b075ff13b9c492ff3345e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a3f7c07b94b075ff13b9c492ff3345e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_91b681264a1dc396b8a85f9f7c6d86fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_955af0d7d2f8decd3b3753b2acefeda5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_fac9c39e262783b876a10f590113891c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6c55bf64bebf6840b097310e9695f75
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6c93aaa23e9e4403f2ad7418bb7d42ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6c55bf64bebf6840b097310e9695f75
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3c471e402869057475e35147e81ee2dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3c471e402869057475e35147e81ee2dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ea2f958ab23678d3e86e78018f917491(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ea2f958ab23678d3e86e78018f917491(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_02abc8b56ffd7a251f33ab5bd2d3d099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_02abc8b56ffd7a251f33ab5bd2d3d099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_eb0f3e5db512755edd64f63275d15c80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_eb0f3e5db512755edd64f63275d15c80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_995589628506760554ef1dd5ed5c37b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_995589628506760554ef1dd5ed5c37b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1feea20828f14924201d6075a338f316(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1feea20828f14924201d6075a338f316(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1f98cd06e79b31aab254343fe4fad531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1f98cd06e79b31aab254343fe4fad531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6248e3b107360867960850fc8ea0040f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6248e3b107360867960850fc8ea0040f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_aea0624b94ffa8c40e3668f653cbb2de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 104, 104], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3943a3a2227fc7aff74de477d97b604f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699eb4e5eb57d0f1079fefbd78a0e769
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 152, 272], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f802de32be79ed5c9872b95286aba4db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 104, 104], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_370f5e8bfb78edec32b40cecb967f58c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_370f5e8bfb78edec32b40cecb967f58c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5f05f102204a270c812d78c664e0348c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([6, 144, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5f05f102204a270c812d78c664e0348c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([6, 144, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e6c46f9fd8d6015cf32ea29a54a2b64f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 184, 184], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_707d73fa0e297c771a933f938bf6b755(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1646cb24811512dd7c2ba042e6b62e8
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_9495c7c3642744c20c83fee342d9353c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1646cb24811512dd7c2ba042e6b62e8
    def get_inputs(self):
        return [
            paddle.uniform([150], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_bf65aec8b3689b3d743361ea82d5074f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bf65aec8b3689b3d743361ea82d5074f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2b320f6263085ddb5d72a73bb136b086(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2b320f6263085ddb5d72a73bb136b086(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d7dc0b32850c19e358cb71bc04583690(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94de098bf59f51bc5f318e663b40eef7
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_74422f787dd0ce997f989feaba953a59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 76, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_370f5e8bfb78edec32b40cecb967f58c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_370f5e8bfb78edec32b40cecb967f58c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_06813718c93b726b879c29420c791008(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ced0d018def9586f4026d70772d35867
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e9b49c9f693b2dc88454dd2941e9f910(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6080f86bc03c604d5d5309d73e5991e3
    def get_inputs(self):
        return [
            paddle.uniform([4204, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_5c74ebe72a133c32c4317096235291f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6080f86bc03c604d5d5309d73e5991e3
    def get_inputs(self):
        return [
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_803f480845fe52a96b02fdefd18ea0c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12d561f68c831a12544831886a48e5bc
    def get_inputs(self):
        return [
            paddle.uniform([4204, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5c74ebe72a133c32c4317096235291f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6080f86bc03c604d5d5309d73e5991e3
    def get_inputs(self):
        return [
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_6b4db25e45f28bf0c1ecea22909cf29e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 256, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_70a35e2a1dcfa62419c784df45e558c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_70a35e2a1dcfa62419c784df45e558c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2741ce497c6c9bce999e73329715a6c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3c471e402869057475e35147e81ee2dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3c471e402869057475e35147e81ee2dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c9d8bd3281eeea09572a4f0b05ce2a8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1646cb24811512dd7c2ba042e6b62e8
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_da95b530f5acb4d9d7255604dd0c1f90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6080f86bc03c604d5d5309d73e5991e3
    def get_inputs(self):
        return [
            paddle.uniform([52, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_3d73f41089552856f69d81517f5339f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1b6cde87f8a67863a073d08d7f0d8017(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1b6cde87f8a67863a073d08d7f0d8017(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ef395d8aaa28832e95a6736c8c92ef64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 92, 92], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bf65aec8b3689b3d743361ea82d5074f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bf65aec8b3689b3d743361ea82d5074f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_aa7ce8b80f5497e4fa63ba64edcea25e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 320], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_556be0f1b171290272ecaf545f4b7be7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([4, 144, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_556be0f1b171290272ecaf545f4b7be7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([4, 144, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_266d8a0e0d3fe4927a2e8e8a54709a32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 20, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_fde0680bbb82e6ddb9e2c1e8eba2e902(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1646cb24811512dd7c2ba042e6b62e8
    def get_inputs(self):
        return [
            paddle.uniform([3800], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_2ceecf0d84daa45e51e82e9cf5761749(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ced0d018def9586f4026d70772d35867
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_adcf53ea5739de892175c6cab1e7ef9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6080f86bc03c604d5d5309d73e5991e3
    def get_inputs(self):
        return [
            paddle.uniform([4680, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_755bbf18f00a88cbc484675aef446b6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6080f86bc03c604d5d5309d73e5991e3
    def get_inputs(self):
        return [
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_45163b39fcc9108481067cef9db3ac4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12d561f68c831a12544831886a48e5bc
    def get_inputs(self):
        return [
            paddle.uniform([4680, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_755bbf18f00a88cbc484675aef446b6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6080f86bc03c604d5d5309d73e5991e3
    def get_inputs(self):
        return [
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_5ea457c5bb1689b4d44bb2512df0049c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5ea457c5bb1689b4d44bb2512df0049c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4e5286c8bfa281b89591ea1c17bfedf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd1fd8aca73a476779e8ace375c8e2a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd1fd8aca73a476779e8ace375c8e2a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1b6cde87f8a67863a073d08d7f0d8017(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1b6cde87f8a67863a073d08d7f0d8017(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5a6ce3e1932ff9aefd6652b2195b8448(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ced0d018def9586f4026d70772d35867
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1c76ae78439f8fb74c60641ff7c73e28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6080f86bc03c604d5d5309d73e5991e3
    def get_inputs(self):
        return [
            paddle.uniform([3778, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_e19bb9e0975b29eb178c6f2d946ceb86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6080f86bc03c604d5d5309d73e5991e3
    def get_inputs(self):
        return [
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_a240a35aa7a313d5123f8ef551b69448(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12d561f68c831a12544831886a48e5bc
    def get_inputs(self):
        return [
            paddle.uniform([3778, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e19bb9e0975b29eb178c6f2d946ceb86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6080f86bc03c604d5d5309d73e5991e3
    def get_inputs(self):
        return [
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_a85f128e03fdb1f07a20e6c01e9cc479(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a85f128e03fdb1f07a20e6c01e9cc479(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ccfcd809da40afe8153c710632c845f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 100], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b46ea491dff479705005a57ac3dcf0ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b46ea491dff479705005a57ac3dcf0ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_063461ffbe0fcbea2a26202de24663aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1174, 1174], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5bb2d0da589e8b609d8a82428a6e147a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_789a1fe3454acae868d05e6627bb54ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 48, 48], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_db06d59f7959e60a34026f53a2a587d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_56a3ec4b4af2067287d7e4d5f3f907aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 38, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_67142b42ef5866f023bf821629b7665d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_80c6b224ada72bb91462d6166c70bb01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_80c6b224ada72bb91462d6166c70bb01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_214fc3da3b3e0414c00e20853395adb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1b0bd44740c92fd3c9c6bb40c1f47fcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 152, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5342c41615eea537ad24a82666b7eaee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([4, 576, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5342c41615eea537ad24a82666b7eaee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([4, 576, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c70ca82c4306523e0ec978c84d7e0f12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8580492930e5e854df65b9a1b7396cd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94de098bf59f51bc5f318e663b40eef7
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 16, 49, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2cb2686098be0d7febb308ee1e1e1d82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_35ceaa5e5fa404197b401080d524f5cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1646cb24811512dd7c2ba042e6b62e8
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_d13acc9d485b95b33ddeca8ebc68b09d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e009cafb56e84a08432cf9dd50c77afa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_80c6b224ada72bb91462d6166c70bb01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_80c6b224ada72bb91462d6166c70bb01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884a142590e018058dfbd87120671753
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()