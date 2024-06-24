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
        return False
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


class TestPrimitiveOp_98d33fdd3f730babc8015e26f3fc5f82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_b21e10151f5f42299c2f2e8d65245bf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 21504, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b21e10151f5f42299c2f2e8d65245bf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 21504, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_072fcc374c8fa38749cbe04545dabad4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e14313a6bee87794a2448baf4666f49b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b2a3a91efb9650f6530fd44288025fc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 92, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d902afe49f83b577c91c678480ffd33f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 152, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_599af9f24a399ef97185fe0362902842(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b9bd72124d43d616e6b2b32691ad92b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_6316a407ed16b16bc2102e42ffd05754(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5ab41128b0e34d6745ef1cbaae1789f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2054f356ba9055a3638bf39961262502(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5abdf7f59b3cd3c79c4470accf3cbfdd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_02f1440ad1c1ae2a5722fcfeec6ca746(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 112, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_8eda3ff0d9bc4b9b5a18fec50de53665(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 20, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.11681878566741943], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_77d086d6e3f831a420213faed2e1ac79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ff1c02be27312f2a345b6e13a70295fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_ba2a59bb137edb21273af43257adc078(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba2a59bb137edb21273af43257adc078(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba2a59bb137edb21273af43257adc078(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3876c74995fe09ab7c7488e9eec80223(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([10, 336, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4e98a4f3b5e3959ecdeb00c7b75f915b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0aaf2290134d34f1cb502ed2c21714d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e98a4f3b5e3959ecdeb00c7b75f915b
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_3605aca3e677045f208df2206e91238b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e98a4f3b5e3959ecdeb00c7b75f915b
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_01421f7dff29dbb58036232da4159fa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e98a4f3b5e3959ecdeb00c7b75f915b
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_579b3061427340ee9bc3774142ea33cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([4, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.38565558195114136]], [[0.15376394987106323]], [[0.29357004165649414]], [[0.4372651278972626]]], dtype='float32').reshape([4, 1, 1]),
        ]


class TestPrimitiveOp_ccc43408a53644f28e0b1fdf2ee9027c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_712ce44f73f445c6f90fad59a0a4b2ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 104, 104], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_6cc05e8ee866ced8824fa896b7b68d7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_15c02e21d8e305d2bdc248bcb0f8246e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26c496af661f594a452b0002b6269924
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 1, 9, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_82eb6736c0600cf9340961549daee5e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa0196bfd5b6c92de0e0cf7b79f08eef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.46808210015296936], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7bdfc8d95d5881032d537aaa3b422cec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7bdfc8d95d5881032d537aaa3b422cec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7bdfc8d95d5881032d537aaa3b422cec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a8d4b9084377e14641249fa70e3237dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.8444730043411255]], [[0.8579772114753723]], [[0.891075611114502]], [[0.8819072842597961]], [[0.8552224040031433]], [[0.8728312849998474]], [[0.8979657888412476]], [[0.9406626224517822]], [[0.8711786270141602]], [[0.8830584287643433]], [[0.8280519843101501]], [[0.827820897102356]], [[0.933842658996582]], [[0.9391326308250427]], [[0.8783131241798401]], [[0.7887889742851257]], [[0.9474055171012878]], [[0.8833380937576294]], [[0.8760688304901123]], [[0.914797306060791]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_9912a4e350a7fec89bd5974abe852560(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cfd8b7ec49adf5771a110066c284a1d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 17, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_168f911defe1aaa3ddc874edfc872b3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9b04847d703598bb74e042f3eee548a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_72e4c9ecb974002ee15b0800eb6246ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_72e4c9ecb974002ee15b0800eb6246ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cdfea2d19b10f4d9188ab25c29cb67b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.2486211657524109]]], dtype='float32').reshape([1, 1, 1]),
        ]


class TestPrimitiveOp_ae7552e214eb8ec216db6c6dc9769586(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2100, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f46e1be56a3016005e1472e1eb1751fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4199826121330261], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c94e0eeae3a50f039ec474da3d8e43a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_44369c558f56569fa5a14473e68165c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d288817f2dffa12fb5b70730be08cd25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([10, 60, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_796c9e4a354976a224ebb08c28f47d67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_289da9e370b87b8230fefc86aef88388(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_289da9e370b87b8230fefc86aef88388(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_289da9e370b87b8230fefc86aef88388(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_289da9e370b87b8230fefc86aef88388(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4f1387dd29e6414fd3ff99c81fb846c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a3c7a027b4b4f54f78609f64a7f9adc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a3c7a027b4b4f54f78609f64a7f9adc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a3c7a027b4b4f54f78609f64a7f9adc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3ad0b708b40dd92c137e5f6029e9e836(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_39a7d1fcc261550cbf04fef66609c933(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_393cc3e4235a3e9da3078af451947413(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ec0b68923ab84191702ffc6976b9bde3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_15e159d930145bdacd667cdabc0dd7fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_acc424a08eccc7671d014e7ee7764423(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc9fef85beb25d55d60313e89f069658(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([150, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([150, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5b380667c745ae07aa6f3c30f9befff2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3543490171432495], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_68b4b5c088d24f26ee69dbfc27008bc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dbe7969787ccd42f4308760d7e77e7ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e44ed1eb463ab82ced519dbf5879859b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([145, 336, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([145, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8bdf773e94530a41ad4b8dcb4ca5a711(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a9607823d82aad624c760d84662d7944(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.26409992575645447], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ca87197b86d2b3191967ad312c6f8026(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 34, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_69a3722aa85569e79e0b83f678c596b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 80, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.7997092604637146]], [[0.8570712208747864]], [[0.8733557462692261]], [[0.6048325896263123]], [[0.8414540886878967]], [[0.7962325811386108]], [[0.9235526919364929]], [[0.9587437510490417]], [[0.8652395009994507]], [[0.823219895362854]], [[0.786716639995575]], [[0.91766357421875]], [[0.7570452094078064]], [[0.8278305530548096]], [[0.800552248954773]], [[0.7880688905715942]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_0e9741420440e75e62d7c52a6dec3537(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_118b37afb46d9ab55a6b167b792772b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([145, 336, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([145, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_53340219ea1673a853e2b40d7bdd17e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4b9bcd38f71b582f84372d9c56252a2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ac16176cbd03e954836a7adf8dbef33c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.0998268723487854, -0.2561909556388855]], [[-0.22796906530857086, 0.2233811914920807]], [[0.34292954206466675, 0.24170681834220886]], [[0.024373650550842285, -0.031865380704402924]], [[-0.06585328280925751, -0.40071728825569153]], [[0.30198371410369873, 0.44445955753326416]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


class TestPrimitiveOp_54a86797219d03e1902c151175005201(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.0926969051361084, 0.14751899242401123]], [[-0.04496400058269501, 0.25635161995887756]], [[0.24204839766025543, -0.03674130141735077]], [[-0.048069506883621216, 0.28515589237213135]], [[0.3324500322341919, -0.46496662497520447]], [[-0.010076262056827545, 0.1572362631559372]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


class TestPrimitiveOp_32bb42d1a2dda63c2a89c4d1e8fdb648(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0998268723487854, -0.2561909556388855]], [[-0.22796906530857086, 0.2233811914920807]], [[0.34292954206466675, 0.24170681834220886]], [[0.024373650550842285, -0.031865380704402924]], [[-0.06585328280925751, -0.40071728825569153]], [[0.30198371410369873, 0.44445955753326416]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.0998268723487854, -0.2561909556388855]], [[-0.22796906530857086, 0.2233811914920807]], [[0.34292954206466675, 0.24170681834220886]], [[0.024373650550842285, -0.031865380704402924]], [[-0.06585328280925751, -0.40071728825569153]], [[0.30198371410369873, 0.44445955753326416]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


class TestPrimitiveOp_ee899e5f10d5656034031d85d56b2a3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0926969051361084, 0.14751899242401123]], [[-0.04496400058269501, 0.25635161995887756]], [[0.24204839766025543, -0.03674130141735077]], [[-0.048069506883621216, 0.28515589237213135]], [[0.3324500322341919, -0.46496662497520447]], [[-0.010076262056827545, 0.1572362631559372]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.0926969051361084, 0.14751899242401123]], [[-0.04496400058269501, 0.25635161995887756]], [[0.24204839766025543, -0.03674130141735077]], [[-0.048069506883621216, 0.28515589237213135]], [[0.3324500322341919, -0.46496662497520447]], [[-0.010076262056827545, 0.1572362631559372]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


class TestPrimitiveOp_53bfb5b3956073ba93a4de313fdfb25f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.020786235108971596], [0.03251347690820694], [0.07385051995515823], [6.456948176492006e-05], [0.06696908921003342], [0.1551518738269806]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([[[0.03387898951768875], [0.08556250482797623], [0.07029067724943161], [0.20596033334732056], [0.11955881863832474], [0.1918669044971466]]], dtype='float32').reshape([1, 6, 1]),
        ]


class TestPrimitiveOp_36c94421e4f0539a4cbf1fe32f4a2910(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.0052885436452925205], [0.01762981154024601], [0.014673925004899502], [0.024182476103305817], [0.186748668551445], [0.003911361563950777]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([[[0.03387898951768875], [0.08556250482797623], [0.07029067724943161], [0.20596033334732056], [0.11955881863832474], [0.1918669044971466]]], dtype='float32').reshape([1, 6, 1]),
        ]


class TestPrimitiveOp_ece4b0380beba92f29f30ec25eb99194(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e659d44b720aecdb8f5d3ee0258bfc84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26c496af661f594a452b0002b6269924
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 1, 49, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4c11da104c44258b60a11370818a98f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 11, 11], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_50d72a027e7b6741fb497fbd6dd31f18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([40, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([40, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a77f52f160feb9878f7fc2d7929463af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7579fa2290cdd01f7fa8eb3d1deff93a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_172c676a1aa62937c04c2180f8b9dade(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9ef45dcd3ae0df98a8bfad37f110b9f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9ef45dcd3ae0df98a8bfad37f110b9f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9ef45dcd3ae0df98a8bfad37f110b9f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9ef45dcd3ae0df98a8bfad37f110b9f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_69452470ff1b9d670f9ffb7aa14b33e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 80, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ead4465b54207a033498dee9f7e5d927(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3800, 81], dtype='float32', min=0, max=0.5),
            paddle.uniform([3800, 81], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_e41a825ac8703c6a541ac5d41cd7503a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([2.1899096965789795, 2.015934705734253, 2.1619725227355957, 2.296905755996704, 2.1067848205566406, 2.177154064178467, 2.085513114929199, 2.127073287963867, 1.9668453931808472, 2.1124396324157715, 2.297358512878418, 2.0862550735473633, 2.051027297973633, 2.205686092376709, 1.9004125595092773, 2.1547632217407227], dtype='float32').reshape([16]),
            paddle.to_tensor([0.9108486175537109, 0.9980205297470093, 0.8823179006576538, 0.5075685977935791, 0.800876259803772, 0.819225549697876, 0.9581784605979919, 0.8103552460670471, 0.6731846332550049, 0.7781937718391418, 0.8941015005111694, 0.9667906761169434, 0.8140385150909424, 0.7879682183265686, 0.9668298959732056, 0.6646597981452942], dtype='float32').reshape([16]),
        ]


class TestPrimitiveOp_bcbf9641a68bacbc8dad01e7aba71b1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([2.1738851070404053, 2.317452907562256, 2.212406635284424, 1.8150107860565186, 1.8200726509094238, 2.085547685623169, 2.192302703857422, 1.9705541133880615, 1.9933676719665527, 1.8663568496704102, 1.8839834928512573, 2.0344581604003906, 1.982454776763916, 2.1147310733795166, 1.8688578605651855, 1.8936680555343628], dtype='float32').reshape([16]),
            paddle.to_tensor([0.08915137499570847, 0.0019794581457972527, 0.11768212914466858, 0.4924314320087433, 0.19912375509738922, 0.1807744801044464, 0.04182153195142746, 0.18964476883411407, 0.3268153667449951, 0.22180624306201935, 0.10589847713708878, 0.03320932760834694, 0.18596151471138, 0.2120317816734314, 0.033170074224472046, 0.3353402018547058], dtype='float32').reshape([16]),
        ]


class TestPrimitiveOp_d12a777f517b1a12a290e503596e7f4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5471202731132507, 0.5041329264640808, 0.5419769287109375, 0.5149013996124268, 0.5124233961105347, 0.5401484966278076, 0.5224947929382324, 0.5243475437164307, 0.4938783049583435, 0.514464259147644, 0.5633956789970398, 0.5211337804794312, 0.5095688700675964, 0.5466001629829407, 0.474841445684433, 0.5168018937110901], dtype='float32').reshape([16]),
            paddle.to_tensor([0.47882625460624695, 0.08157579600811005, 0.30465835332870483, 0.43199852108955383, 0.33976075053215027, 0.10316277295351028, 0.37339144945144653, 0.15166468918323517, 0.14039954543113708, 0.09372878074645996, 0.2571961283683777, 0.17736618220806122, 0.331104576587677, 0.012373730540275574, 0.2723919451236725, 0.0932813212275505], dtype='float32').reshape([16]),
        ]


class TestPrimitiveOp_5b537f1c46b1ff993ab69c58be93406c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([145, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([145, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_849bad304e4661f424b5cf25d17dab8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 80, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4818625748157501], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14b0cf5c51d8f980138a1fa528d0306f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_da79dd60741bbd1567e79af19c32f84c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_da79dd60741bbd1567e79af19c32f84c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_da79dd60741bbd1567e79af19c32f84c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a67f3c83a3536cfc750eccf57f7e841f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 14, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.15895339846611023], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c505d65b94cb4591275677ff004a1c3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.10348998755216599, 0.044277485460042953, 0.21761225163936615, 0.07678458839654922]]], dtype='float32').reshape([1, 1, 4]),
        ]


class TestPrimitiveOp_ba7cdd62c136f6f8767a9f93b188d9dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4b66509dcbfc576044ae90fefbd2aae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2630232274532318], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_80619df623e36111be5c9e4048096493(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.011396851390600204], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cebe411f4938519a40ab8eb9e0e2b985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.322111040353775], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ba2a59bb137edb21273af43257adc078(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba2a59bb137edb21273af43257adc078(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba2a59bb137edb21273af43257adc078(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba2a59bb137edb21273af43257adc078(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae180e04ef411d9166814961f037b343(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae180e04ef411d9166814961f037b343(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae180e04ef411d9166814961f037b343(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d67e460d4853bede291f68355781a239(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 34, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0267185a635dd0ccf9b40cbee69343cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([3, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.4750056862831116]], [[0.3209798336029053]], [[0.2568384110927582]]], dtype='float32').reshape([3, 1, 1]),
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


class TestPrimitiveOp_c4e4e42515dd01165b40041b3b394118(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef905cefeca655659c06d8284090abaf
    def get_inputs(self):
        return [
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b927250f1ed493ebd521e2f1cd143a6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([150], dtype='float32', min=0, max=0.5),
            paddle.uniform([150], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c98e45753e06d178b925dd5030bd4538(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([22, 60, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2054f356ba9055a3638bf39961262502(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_301226d723a466b2dc2aefd19daabfbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.31504109501838684], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_84ac7e020851a204410d683fe864077c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fb0465d663fbc54d4952049367ea58ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f5b0367a903ac0018752e2414801347f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_85ed5c39e91917f88b7a4665bab2673c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd97aa78fd5322c7b85710e4b6f0eb2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_159c002927a2ca6ed0c6dace31dabed2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 872, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 872, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5868663be13fae4fb58dbb4084b94ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5868663be13fae4fb58dbb4084b94ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5868663be13fae4fb58dbb4084b94ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5868663be13fae4fb58dbb4084b94ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c0f4a9030ca4eda120d11970470b86f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d0d872325da5ee262c3a60f91f6c7f9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc6c0a641f486519e5c1a5d46c0ff81b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc6c0a641f486519e5c1a5d46c0ff81b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc6c0a641f486519e5c1a5d46c0ff81b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc6c0a641f486519e5c1a5d46c0ff81b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc6c0a641f486519e5c1a5d46c0ff81b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75c33f66e178e46c9c7a3f06f60232e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1745, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1745, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75c33f66e178e46c9c7a3f06f60232e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1745, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1745, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc6c0a641f486519e5c1a5d46c0ff81b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1fe14a1ada2bb144a12326ab62490cc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_edaaf85d4fa14fb8fcdf25538b26a455(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 112, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]]], [[[1.0]]], [[[0.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[0.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_d965fb0cfd7f59d542b710f3df1fb712(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.04129378870129585], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_290224ec9eaaa6694d32cb628498fe08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d595ccd219ff42be46f971738d166d76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 40, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[0.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_25ef4f31d3f3ad449b5af0791f443090(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3473ffcf08d723e0ae3fe245db348b5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.43115514516830444], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7a5f25bc8715456f1e15dbeb1408950c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26c496af661f594a452b0002b6269924
    def get_inputs(self):
        return [
            paddle.uniform([22, 4, 1, 49, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1f0050d3cbc1aef0b73f7bc36c6e49a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1f0050d3cbc1aef0b73f7bc36c6e49a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1f0050d3cbc1aef0b73f7bc36c6e49a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8c700c32d13ac44b6a0488c6b78025c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8c700c32d13ac44b6a0488c6b78025c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8c700c32d13ac44b6a0488c6b78025c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8c700c32d13ac44b6a0488c6b78025c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99f45001335888be4d29b1545cf615c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f63990b79f2a7471c839781b29c0ff59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([171, 336, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([171, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_239c87785e8dbc0a37fad7a501499d98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_23a6eb4db8f360c8cdf32b5f1631c431(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 24, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_935f5ff84f85ab1b6b00d28f0bed42f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7b1d3ae47d9d5b2c504d319d83cf80f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 11, 11], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_212f070da2864c491503ef25fdcb7931(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8bf3088cc16f31c965831062ca57f3f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8bf3088cc16f31c965831062ca57f3f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8bf3088cc16f31c965831062ca57f3f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f537f9aa64f880cf4822312fd928b129(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 96, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dbef3b299bbb92e6be36d6c1f22470b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba4bbdd54b2066d9e366beb87acc6809(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4a2605b3435a2b4b649f4a5cfc03b38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4a2605b3435a2b4b649f4a5cfc03b38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4a2605b3435a2b4b649f4a5cfc03b38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4a2605b3435a2b4b649f4a5cfc03b38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5421c9ce25248b8e3e2a18a7f3146364(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_59735b0117ac37adae0c87f9b6e4aecb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.009430937469005585], [0.0], [0.0], [0.29372572898864746], [0.0], [0.0], [0.07399353384971619], [0.0]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.09906633198261261], [0.0], [0.0], [0.0], [0.12775105237960815], [0.0], [0.0], [0.10675060749053955], [0.015985265374183655]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_311068a7fa86c312192badbb948b1bdd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.15899638831615448], [0.009430937469005585], [-0.012757033109664917], [0.17623372375965118], [0.310524046421051], [0.3362637758255005], [0.19883430004119873], [0.07399353384971619], [0.25101494789123535]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.15629974007606506], [0.09600900113582611], [-0.26919764280319214], [-0.18770547211170197], [0.21689829230308533], [-0.06884871423244476], [0.19098523259162903], [0.10675060749053955], [0.01886458694934845]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_27f71b0ff944870678ee5161224626e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.1744346171617508], [0.28419965505599976], [-0.07115548849105835], [-0.15713690221309662], [0.34863829612731934], [-0.032754749059677124], [-0.2847267985343933], [0.4185256063938141], [-0.21497729420661926]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.3240494132041931], [-0.19642935693264008], [0.19043558835983276], [-0.009380102157592773], [0.2472778558731079], [0.3632349371910095], [0.03395096957683563], [0.44917553663253784], [0.07401689887046814]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_23ad00c2c14c3bbb3bf1bb52af4c8009(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.11072088778018951], [0.28419965505599976], [0.20599952340126038], [0.17623372375965118], [0.3654366433620453], [0.3362637758255005], [0.19883430004119873], [0.4185256063938141], [0.25101494789123535]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.38128283619880676], [0.13261474668979645], [0.19043558835983276], [-0.009380102157592773], [0.3364250957965851], [0.3632349371910095], [0.2574111819267273], [0.44917553663253784], [0.07689622044563293]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_4863487f2364636bcb73a95e986e1d2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_29da357f6c59fb4a53d51407545de8e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([10, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f2d25e44fcd5823b98187b408f9d56b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_290eeb805aa7ea9fc539822d9e39bcea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92495f614a6f71af2a2483214392eb02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba00d8c3445bdd80e5c7014ea3c2de0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.02323956787586212], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ba00d8c3445bdd80e5c7014ea3c2de0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.02323956787586212], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_367861a4667d5e4cad49e7f5d05a1e54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_73d587f74fdf57dba967756e9edbce51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.368469774723053], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_daf91d53271189d24768ca5ed1c498e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f5b693bfcd64ce3b8aca5a8d8b772abd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9838f5370f876b25bf8cdb571b693f8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3b592bfe77464498888aa0cf5e1050da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3b592bfe77464498888aa0cf5e1050da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3b592bfe77464498888aa0cf5e1050da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3b592bfe77464498888aa0cf5e1050da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3b592bfe77464498888aa0cf5e1050da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_786fb9ec8c15525a250f938d7671a23a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([5556, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5556, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_786fb9ec8c15525a250f938d7671a23a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([5556, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5556, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3b592bfe77464498888aa0cf5e1050da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_313bd69d0a61647b7968e06331a9f959(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bce34f3a3338a7f037f8b922ea0fb2ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_106205bf98f6c0b034d4c427c49b2b56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c07474e27d158ce947768c05ff5320e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c07474e27d158ce947768c05ff5320e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c07474e27d158ce947768c05ff5320e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7c92877656aab07b4a78b1b316e33a7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e73aa0c0b0d467a36351d66098fe85f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.05760360136628151], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_19f981dac93e722593aa92a466b03b91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 17, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc05ac48703bf6e773f90e05c66f6a57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fa43458f646735ff743f0aa76c928d43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_991ad62d9aa1d2eba4f0d1dc1577ff5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([10, 336, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8dda477bbf7abddec9bbfba452aa425a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e9741420440e75e62d7c52a6dec3537(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ad3c4b39b065491d0ee778498834c5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 88, 88], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_360cd11b4f696e2b78fb2c78d9a6f46a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([15200, 81], dtype='float32', min=0, max=0.5),
            paddle.uniform([15200, 81], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eb025f2fd4412b8ad554ec37d7d90470(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 160, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a6c37ad32745fc2dbdfcc189fc67183(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c3a90a6477db2298a5aebf5ba17db9ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c03d42530b291fd867b4eb20498269dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 10, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4141620099544525], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1bc2c1aee291fd019890c8e11d7279b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([15200, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([15200, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_42da361ca79276838cea2e6ada7e2ae5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.6211273074150085], dtype='float32').reshape([1]),
            paddle.to_tensor([0.38200703263282776], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ff6ab491adc4c607be52ec1a6dc45b0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.9330731630325317], dtype='float32').reshape([1]),
            paddle.to_tensor([0.46027541160583496], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4c00802232d357dd8798f040abbac529(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26c496af661f594a452b0002b6269924
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 1, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6775c04808c01464721b7af4f69590c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_318112fbb3a78cef01b64ba6ced6256d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a8996538e6183c5c5309d6da2d7f3fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d0663d6b707f9002f4a8da5a38eea75b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 168, 168], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e6b5ca59d3cad022450fdbb6409f3afc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d39225b310187c5d30127935ec266e15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([10, 36, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e65360242af7087a3ab1aca65b5a1953(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.15387283265590668], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_393cc3e4235a3e9da3078af451947413(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_35665140f17efa04cc018a113c05adbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.48166215419769287], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e6edf2fa365d49db023e2379419b69bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5421c9ce25248b8e3e2a18a7f3146364(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8a5bda07c8c5d72e5eb320ff277e286f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8e26bc8310361d8e14197d930b18f716(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8d4c32d15aefa1cc3e84a8f0279ccebe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 0.02797788381576538, 0.12474798411130905, 0.0, -0.2654902935028076, 0.19371631741523743], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.1567399799823761, -0.03549548238515854, -0.1307554543018341, -0.10650692880153656, 0.0, -0.13101938366889954], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_1cb6685de386137fca5651f314a3935d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.0, -0.0009930884698405862, -0.01631147973239422, -0.0, -0.0, -0.025380592793226242], dtype='float32').reshape([6]),
            paddle.to_tensor([0.0, 1.0, 1.0, 0.0, 0.0, 1.0], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_3b2ce858dea53b026e14477cc47a600b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.0, -0.0009930884698405862, -0.01631147973239422, -0.0, -0.0, -0.025380592793226242], dtype='float32').reshape([6]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_02c4cfa830312a587e101140a99ac525(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 0.02797788381576538, 0.44838252663612366, 0.0, 0.0, 0.3485032021999359], dtype='float32').reshape([6]),
            paddle.to_tensor([0.02018606662750244, 0.3696957528591156, 0.0, 0.12233604490756989, 0.0, 0.0], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_1774f1eeee6f7361e0ddf2e6018c46a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.052370764315128326, 0.19320246577262878, 0.1291169971227646, 0.3269874155521393, -0.2654902935028076, 0.19371631741523743], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.12136902660131454, -0.03549548238515854, 0.15166980028152466, -0.10650692880153656, 0.136802077293396, 0.0007996018975973129], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_86fa0001b2f6cbe1e0997a19336ecc76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.02075279876589775, 0.02873837947845459, 0.16400176286697388, -0.029398426413536072, -0.02302578091621399, -0.03884989023208618], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.02075279876589775, 0.02873837947845459, 0.16400176286697388, -0.029398426413536072, -0.02302578091621399, -0.03884989023208618], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_ea4ed3105e7fc29e734c43c5d0687cc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10614849627017975, 0.19129973649978638, 0.07027313113212585, 0.04235222935676575, 0.027911990880966187, 0.10930299013853073], dtype='float32').reshape([6]),
            paddle.to_tensor([0.10614849627017975, 0.19129973649978638, 0.07027313113212585, 0.04235222935676575, 0.027911990880966187, 0.10930299013853073], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_7fbda59f6e56341b6e3f2eb5d2f24082(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.052370764315128326, 0.19320246577262878, 0.4527515172958374, 0.3269874155521393, 0.0, 0.3485032021999359], dtype='float32').reshape([6]),
            paddle.to_tensor([0.052370764315128326, 0.19320246577262878, 0.4527515172958374, 0.3269874155521393, 0.0, 0.3485032021999359], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_1cf8fa3b7bad08e65243a441da02ad50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0555570125579834, 0.3696957528591156, 0.28242525458335876, 0.12233604490756989, 0.136802077293396, 0.1318189799785614], dtype='float32').reshape([6]),
            paddle.to_tensor([0.0555570125579834, 0.3696957528591156, 0.28242525458335876, 0.12233604490756989, 0.136802077293396, 0.1318189799785614], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_339d0ed9b4e75825f78e85f578ffab44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.12757647037506104, -0.5935946106910706, 0.7010362148284912, -0.4034132659435272, -0.6893715262413025, 1.2204018831253052], dtype='float32').reshape([6]),
            paddle.to_tensor([0.31478211283683777, -1.4646350145339966, 1.729736328125, -0.9953817129135132, -1.7009549140930176, 3.011219024658203], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_2947d2a1978a2cc231948bb88af46e3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.03860832750797272, 0.46506884694099426, 0.5480446219444275, 0.286504328250885, 0.5397198796272278, 0.7860916256904602], dtype='float32').reshape([6]),
            paddle.to_tensor([0.04015878960490227, 0.8693994283676147, 1.2126078605651855, 0.40155017375946045, 1.172589898109436, 3.6748974323272705], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_68c9ba7fe86a2acfb622246dc272eabe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([10, 480, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b8e0f2fff3ff5c2e0fe386a25ebdc7f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c57bbb072289b66cd96ae8a83bf14aa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_787af3a9ce59014f5707bc862eee09c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_787af3a9ce59014f5707bc862eee09c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_787af3a9ce59014f5707bc862eee09c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_787af3a9ce59014f5707bc862eee09c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_787af3a9ce59014f5707bc862eee09c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d5fc8fd17055531663a1e5eac617b9da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1744, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1744, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d5fc8fd17055531663a1e5eac617b9da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1744, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1744, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_787af3a9ce59014f5707bc862eee09c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9838f5370f876b25bf8cdb571b693f8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b2d03a0c8272820e6457e4ec31da06a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae0d8f668cad9d84244b0e334231f974(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 256, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb1d106f6f19620bb7d34de1e4da6ccb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([22, 336, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1fe14a1ada2bb144a12326ab62490cc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b8aea1a32ae1eddece40e34551a4ecd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_97e72ca07c6dfad8124fb31fc3d32494(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 76, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_89a2e08d6256d7f7b66c641241c2b24c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.060890283435583115], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ead8547a0bdcdd08caced02fd636d11f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d40a43751e4a6557ffb214ee9d974de1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_29f00098c1714dbb564378c6a51cac97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba7cdd62c136f6f8767a9f93b188d9dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4deb12f6278f134b74fdc928caa4db7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([171, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([171, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b00ddbbab25548c4c0621486fe76d16e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([171, 336, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([171, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9282ea0696621093bf67c290e6509ab4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 480, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_818c17e43569c0d72bd213274ccf97e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9ea6dbafd2326cdda518d995f14ccdae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0a54144fb1c6cadd4cf5e7ae946e8150(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.03314857557415962], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_74ae9aa376d4ef0d00838ad64be99d3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26c496af661f594a452b0002b6269924
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 1, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cf6ca2f6a8f4896dbe08c9946927385d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([1.9989869594573975, 2.234851360321045, 1.9859070777893066, 2.2829952239990234, 1.8882437944412231, 2.129962682723999, 2.2910149097442627, 2.1324944496154785, 2.0255579948425293, 2.0232930183410645, 2.035919427871704, 2.2206926345825195, 2.130520820617676, 2.0110466480255127, 2.173865556716919, 2.009289026260376, 2.2680442333221436, 2.2394838333129883, 2.2760093212127686, 2.1627626419067383, 2.1290602684020996, 2.2944893836975098, 2.118405818939209, 2.238065242767334], dtype='float32').reshape([24]),
            paddle.to_tensor([0.9421505331993103, 0.5644310116767883, 0.5687127113342285, 0.5983794927597046, 0.5380127429962158, 0.9941259622573853, 0.8361309766769409, 0.6516550779342651, 0.6754908561706543, 0.6397345066070557, 0.9202067255973816, 0.9728923439979553, 0.9369902610778809, 0.8909742832183838, 0.7103930711746216, 0.8623802065849304, 0.681532621383667, 0.6159887313842773, 0.5615475177764893, 0.832107663154602, 0.8940337896347046, 0.6315939426422119, 0.7134082913398743, 0.7348434925079346], dtype='float32').reshape([24]),
        ]


class TestPrimitiveOp_306b9beff54c3526826ee55ef222c246(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([2.222540855407715, 2.1537089347839355, 2.120952844619751, 2.2830405235290527, 1.9556010961532593, 2.3045382499694824, 1.898406744003296, 2.294344663619995, 1.8883681297302246, 1.9440796375274658, 1.9587922096252441, 1.8760839700698853, 1.9374864101409912, 2.1575067043304443, 1.9762415885925293, 2.2733170986175537, 1.890726089477539, 2.019907236099243, 1.9900991916656494, 1.9032824039459229, 1.9374114274978638, 2.0630862712860107, 2.055297374725342, 1.841697335243225], dtype='float32').reshape([24]),
            paddle.to_tensor([0.05784948170185089, 0.43556898832321167, 0.4312872588634491, 0.4016205370426178, 0.46198728680610657, 0.0058740247040987015, 0.16386902332305908, 0.3483448922634125, 0.3245091736316681, 0.36026546359062195, 0.07979327440261841, 0.02710764855146408, 0.06300970911979675, 0.109025739133358, 0.2896069288253784, 0.13761979341506958, 0.318467378616333, 0.38401126861572266, 0.43845245242118835, 0.16789230704307556, 0.10596621036529541, 0.3684060275554657, 0.28659170866012573, 0.26515650749206543], dtype='float32').reshape([24]),
        ]


class TestPrimitiveOp_1a50026b6d260def3a95c776c3c83694(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5029798746109009, 0.5498770475387573, 0.5110376477241516, 0.5707533955574036, 0.4798405170440674, 0.5327470302581787, 0.5566696524620056, 0.5472185015678406, 0.49525970220565796, 0.49868878722190857, 0.5074412822723389, 0.5528377294540405, 0.5295894145965576, 0.50675368309021, 0.5291580557823181, 0.5114061236381531, 0.5369701385498047, 0.5387910008430481, 0.5376628041267395, 0.5297994613647461, 0.5271880030632019, 0.5523097515106201, 0.5250798463821411, 0.5332414507865906], dtype='float32').reshape([24]),
            paddle.to_tensor([0.39899784326553345, 0.28851377964019775, 0.32047808170318604, 0.1115952581167221, 0.22549396753311157, 0.3493143320083618, 0.01095319353044033, 0.32153812050819397, 0.4632074534893036, 0.21500954031944275, 0.1573912650346756, 0.23264914751052856, 0.29595470428466797, 0.3311556279659271, 0.13709194958209991, 0.41734373569488525, 0.10689177364110947, 0.17011195421218872, 0.4864208698272705, 0.046979255974292755, 0.11242204904556274, 0.35655316710472107, 0.0788332000374794, 0.18212632834911346], dtype='float32').reshape([24]),
        ]


class TestPrimitiveOp_5c4457e83e8357111f0f700039a330e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([171, 60, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([171, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0aaf2290134d34f1cb502ed2c21714d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e98a4f3b5e3959ecdeb00c7b75f915b
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_3605aca3e677045f208df2206e91238b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e98a4f3b5e3959ecdeb00c7b75f915b
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_01421f7dff29dbb58036232da4159fa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e98a4f3b5e3959ecdeb00c7b75f915b
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_a3c7a027b4b4f54f78609f64a7f9adc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a3c7a027b4b4f54f78609f64a7f9adc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a3c7a027b4b4f54f78609f64a7f9adc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a3c7a027b4b4f54f78609f64a7f9adc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8c700c32d13ac44b6a0488c6b78025c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8c700c32d13ac44b6a0488c6b78025c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8c700c32d13ac44b6a0488c6b78025c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_831ab6023b0c01a367d14f045c059050(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_831ab6023b0c01a367d14f045c059050(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_831ab6023b0c01a367d14f045c059050(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_831ab6023b0c01a367d14f045c059050(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_831ab6023b0c01a367d14f045c059050(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_46df11eecbbb67f3e63c0329a197224b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1547, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1547, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_46df11eecbbb67f3e63c0329a197224b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1547, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1547, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_831ab6023b0c01a367d14f045c059050(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9794fdf6fb3d3d5ff6d5c5f8cd521f6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9794fdf6fb3d3d5ff6d5c5f8cd521f6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ac278c7572965360db6d2d192ff2b997(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.24266022443771362], [0.24176530539989471]]], dtype='float32').reshape([1, 2, 1]),
        ]


class TestPrimitiveOp_71d0ad794e64ebe11ee559c5b8f8a1f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3549, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_887af87087a46ee5c6f7dae7f77b322d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26c496af661f594a452b0002b6269924
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 1, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_68e485f27caeaad4d4c0e07a07c9e95a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_da79dd60741bbd1567e79af19c32f84c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_da79dd60741bbd1567e79af19c32f84c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_da79dd60741bbd1567e79af19c32f84c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_da79dd60741bbd1567e79af19c32f84c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3c7ea682e5c7ab1880d15e7ee4c4e123(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 136, 136], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ece4b0380beba92f29f30ec25eb99194(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_052e4b474eff6d7cc42e3bcfc07ad100(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dbf0d4cfd1bfaebe282af17492b83aa4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.07177720963954926], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cd4287e6803388ee01815b63144978f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 76, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9de8c85db8c7ec751bbbc741911f5148(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4199f19ba8db282163c24b68d4111789(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([2.0487561225891113, 1.9004993438720703, 2.2151260375976562, 2.1765787601470947], dtype='float32').reshape([4]),
            paddle.to_tensor([0.6998443603515625, 0.825813889503479, 0.7969165444374084, 0.7159750461578369], dtype='float32').reshape([4]),
        ]


class TestPrimitiveOp_5f40d8d9323bd09bed22065f0a06d66c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([2.1404459476470947, 2.0547609329223633, 2.1407315731048584, 2.1194119453430176], dtype='float32').reshape([4]),
            paddle.to_tensor([0.3001556098461151, 0.17418614029884338, 0.20308345556259155, 0.2840249538421631], dtype='float32').reshape([4]),
        ]


class TestPrimitiveOp_ea8f8578c68008fdbcefb2869a7a2c53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5190693140029907, 0.48184239864349365, 0.5500044226646423, 0.54008549451828], dtype='float32').reshape([4]),
            paddle.to_tensor([0.46462282538414, 0.45306992530822754, 0.2873522639274597, 0.28311678767204285], dtype='float32').reshape([4]),
        ]


class TestPrimitiveOp_7bdfc8d95d5881032d537aaa3b422cec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7bdfc8d95d5881032d537aaa3b422cec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7bdfc8d95d5881032d537aaa3b422cec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7bdfc8d95d5881032d537aaa3b422cec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b99739170eb358112e7a8e9777baa793(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb21c8f8298c616898209a396c2cd3d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 80, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cf2b636a6b74ffca4b45cc9bed73b70a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7fc656ab4bc876ef9fd52ddba6c82399(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2204, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([2204, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8e26bc8310361d8e14197d930b18f716(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ee593dcc23494f99a81a323dd2ad95c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([22, 36, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 36, 1, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_fa2387eff9ea1776cda0d2e1dde28fd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7cbf9ea34c096a043368dc76a2781b7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.21656160056591034], dtype='float32').reshape([1]),
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b073031ef0b6591773e92427bc7d210f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_365aaf9caff88d7c59617b93806ee263(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.32400184869766235]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.27040180563926697]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_0327cd70667f7e85856b691ec521a063(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.1259002685546875]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.004428437910974026]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_9974a39e04e8c49066fe9a06a2e8a5ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.32400184869766235]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.46344733238220215]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_887af87087a46ee5c6f7dae7f77b322d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26c496af661f594a452b0002b6269924
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 1, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_88b924bfee3e08da1d45da2ba34c3130(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.011999249458312988]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.11023341119289398], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_1bbae809b740b558f0ac53b3fa79d528(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.12224304676055908], [-0.13875849545001984], [0.3131491243839264], [0.15301255881786346], [-0.15440522134304047], [0.011999249458312988]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.11671420931816101], [0.06449782848358154], [0.06832995265722275], [0.38540637493133545], [-0.004189184866845608], [-0.13240405917167664]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_408f2fa3967b849472ed9bc58b215ccd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.22099769115447998], [-0.10152854770421982], [-0.3057101368904114], [-0.18587660789489746], [0.08716388791799545], [0.3550001084804535]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.38683974742889404], [-0.06397268176078796], [-0.0968986302614212], [0.11023341119289398], [-0.4506143629550934], [-0.13261719048023224]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_cc7577977f6117f753c5894c701ccc7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.22099769115447998], [-0.0774497240781784], [0.3131491243839264], [0.15301255881786346], [0.18806904554367065], [0.3550001084804535]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.11671420931816101], [0.06449782848358154], [0.0786060243844986], [0.38540637493133545], [-0.0031212486792355776], [0.037856653332710266]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_918a8188ec99ce1d20754accdfc0e491(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_38ceb4e2f17524ea4edf6d49a424558b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd4dfdd640e600c295534a1fe10eec3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d6734790080d37c14106b5026fde07c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 24, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_8a30ef612e00cc654d98a4fe30ffb042(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 104, 104], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_876bd997c1d43514297ae980fcc63e6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 184, 184], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_18f9de504038d060dc03ae62506f3537(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.737528383731842]], [[0.9224912524223328]], [[0.721898078918457]], [[0.8820279836654663]], [[0.7744972705841064]], [[1.0]], [[0.8276877999305725]], [[0.7421025037765503]], [[0.7160605192184448]], [[0.9357027411460876]], [[0.7740103006362915]], [[0.9700562357902527]], [[0.8731153607368469]], [[0.7251651287078857]], [[0.7410420775413513]], [[0.8984760046005249]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_8f8530773a84e929804f8c2063287356(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dbde90ffb8111a1b5b8318912fbc225a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_226a11f231ee6cb7b43823418027f566(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4be926f783e9cebfa74b3c2eb4a97e21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([145, 60, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([145, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4e1f44e06e2cc6b4982d58915c2acf97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3680667281150818], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_182b18854d9d5d48edb7aa769e537461(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([70, 81], dtype='float32', min=0, max=0.5),
            paddle.uniform([70, 81], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a6c1e659086d5b01e6f0e9af14df8cdd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b5514196fe16e168c00cdf88b5d013cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b5514196fe16e168c00cdf88b5d013cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c106fd06a6a213831e6f7051e15ea3d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.24498169124126434]]], dtype='float32').reshape([1, 1, 1]),
        ]


class TestPrimitiveOp_0a7edb54bfa9debf96566555e4b868e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4116, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_629d020e682a5e07802d6c2447847082(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([551, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([551, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f0af628c5c63b22e5e6d2d9a8fe1aa7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2c7e0c2bd49d5c146c844f745bc87662(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([22, 336, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f3ee434ba12731eef495d58d467e7f30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f8b47025e59de02b1a71c017408220c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1f0050d3cbc1aef0b73f7bc36c6e49a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1f0050d3cbc1aef0b73f7bc36c6e49a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1f0050d3cbc1aef0b73f7bc36c6e49a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1f0050d3cbc1aef0b73f7bc36c6e49a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ccc43408a53644f28e0b1fdf2ee9027c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_60d48e17027d8d86550381922c9007f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 9, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a4ca02e6581071903c6a43ee094ab592(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bf1fdfa56a416da421417924b802250b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([247, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([247, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c0c664b71c65e0d09e9f0034323206d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_caa1fdce50a9e5f2781244dff429ad37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 88, 88], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7c9daa431e23a50d12db08cee6406e40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c6f36548c9a25cc4b1ce5f1d53c4eac6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 104, 104], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e9351a4f0137abe0d9150fddbac4ad3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([950, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([950, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d6da5957b36ee63cbdfae0f73863728(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_74ae9aa376d4ef0d00838ad64be99d3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26c496af661f594a452b0002b6269924
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 1, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f4eee5533aaa16e1147bad77d5b735b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e623bfe4b9f4c2ab95d1a9a207a9757d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_289da9e370b87b8230fefc86aef88388(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_289da9e370b87b8230fefc86aef88388(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_289da9e370b87b8230fefc86aef88388(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cd85949fd221f80d58fb36c7f6ccc808(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae7b95295dffa7c524f7eeda6ba0e43d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cf8cf1235a671b9af4e3d1d1be123d57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d144854f3da7f05ba6d2ea88de08fa6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5e1255feb90c12762e7ce8a6433e68e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4189532995223999], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e565ee2acf9ee57916d7a11375fba47c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([950], dtype='float32', min=0, max=0.5),
            paddle.uniform([950], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3b10475a6e6cc0c52542e48cca9654ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8f8ad2d9b5aa69508997838667bff502(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ab7b1893b50be5a1f2a33e7b95ddccaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([8816], dtype='float32', min=0, max=0.5),
            paddle.uniform([8816], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d6ea463dc6bcd2896e3ebda7c254fe4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d6ea463dc6bcd2896e3ebda7c254fe4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d6ea463dc6bcd2896e3ebda7c254fe4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d6ea463dc6bcd2896e3ebda7c254fe4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d6ea463dc6bcd2896e3ebda7c254fe4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4de9f3232561f1a8218c44251cd85658(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2056, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2056, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4de9f3232561f1a8218c44251cd85658(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2056, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2056, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d6ea463dc6bcd2896e3ebda7c254fe4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8922224a94cb1a6b9f55f06ea1f76f64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b78345cab29053b1c6bf77823641eb70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.8192763924598694], dtype='float32').reshape([1]),
            paddle.to_tensor([0.04321257397532463], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_208647f989163b8609d4f94ab9d2531a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.9048857688903809], dtype='float32').reshape([1]),
            paddle.to_tensor([0.3429238200187683], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d5a9af990b7fbe404f6a59d60355636f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.6332545876502991], dtype='float32').reshape([1]),
            paddle.to_tensor([0.15142355859279633], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9e786b4c2e8d2d805ea1ed32ffdb8ecb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.7331942915916443], dtype='float32').reshape([1]),
            paddle.to_tensor([0.38929012417793274], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_47533a28af8bd170a1802ca88c652bea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.9347270727157593], dtype='float32').reshape([1]),
            paddle.to_tensor([0.03657282516360283], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0fe4a67b386fabf8c3d817422d9d49e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.8455761671066284], dtype='float32').reshape([1]),
            paddle.to_tensor([0.24542270600795746], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a0ba8b257c83da61d50fd2e469ef0a79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.8574271202087402], dtype='float32').reshape([1]),
            paddle.to_tensor([0.4412666857242584], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_45ddcc80fdb54e3186a774154ac344c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.7159499526023865], dtype='float32').reshape([1]),
            paddle.to_tensor([0.07872757315635681], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_027f94ed3086744160d9e75d5fcdeb55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.8316724896430969], dtype='float32').reshape([1]),
            paddle.to_tensor([0.38323476910591125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f0a705b2fb221cf48033609b81cf979e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([8816, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([8816, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f2418ad7e7ede2ae716ce43afa2d876f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f2418ad7e7ede2ae716ce43afa2d876f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f2418ad7e7ede2ae716ce43afa2d876f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f2418ad7e7ede2ae716ce43afa2d876f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d66501acc8efb60a1eb4900db994d46e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e50d75fe94baddf93b093e922d2df637(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 42, 42], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_34a98c6aa7e8a6a239d4509fedb8c179(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_761e6896ba2d2ceefe143ac904f6d0e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.34753337502479553], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_761e6896ba2d2ceefe143ac904f6d0e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.34753337502479553], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e526be2594c8b1b5d7b6373d2b479bc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_93b069772dc6017bd88bbee9acb977a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2228458672761917], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_044dc86321bceff4e2cff133db6c9b21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91fd2d4436f2ebf5c122f4a90e2d7d07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ffb065c3dd5fd6d52de198228055ba50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ffb065c3dd5fd6d52de198228055ba50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ffb065c3dd5fd6d52de198228055ba50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ffb065c3dd5fd6d52de198228055ba50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ffb065c3dd5fd6d52de198228055ba50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_184937177d52c11017e7c179ce2215ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([4650, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4650, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_184937177d52c11017e7c179ce2215ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([4650, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4650, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ffb065c3dd5fd6d52de198228055ba50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6775c04808c01464721b7af4f69590c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e8c207ed8b74b4e2e0fe830258492eee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([6, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.49028000235557556]], [[0.40219151973724365]], [[0.0392024964094162]], [[0.1044401228427887]], [[0.0016230230685323477]], [[0.3070937693119049]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_a5274fc66d1a48c64a70c768fe881708(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_705bbf60f84a42ce70b86b84727843b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_705bbf60f84a42ce70b86b84727843b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_705bbf60f84a42ce70b86b84727843b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_705bbf60f84a42ce70b86b84727843b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_705bbf60f84a42ce70b86b84727843b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4d738d91f2913e6b194aa29d4b7c2aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1059, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1059, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4d738d91f2913e6b194aa29d4b7c2aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1059, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1059, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_705bbf60f84a42ce70b86b84727843b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c341af0e76cde65225ec5a5c688492b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_692fffa4be6db28c218cec3c793f8ef4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_51788daa59f2f3981f0eb5166c741a3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f3ee851c26bc4625679004db9bf57ff7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_915a2e29fd1528694e9294df47f7914e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f67bd2996c63cd18d10d457d6cfaa19a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e635e1a0f98f2ffa90ad0ad85c23b4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c526557767e6690fbc4fd409e3579f39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_921fb9fcda2279ca90a0b49c36cbbb24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b8aea1a32ae1eddece40e34551a4ecd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d4490649479a3bfcbdaee4642b5bc04c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd4dfdd640e600c295534a1fe10eec3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_85b8d6c29dad36e811a3279c00bb66df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c7315cb3e19f3efeac0d1e27fab853b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9ef45dcd3ae0df98a8bfad37f110b9f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9ef45dcd3ae0df98a8bfad37f110b9f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9ef45dcd3ae0df98a8bfad37f110b9f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_efd0efa255c5e5e1d126ad3f4cb0555f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_adcce674fc5ad8c69eb49dc1f9b65cb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_adcce674fc5ad8c69eb49dc1f9b65cb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a3e58a9ea40015a595c2d7af88515c52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.015143156051635742], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e81e2cde8a4b85dbf95f95ae1c1814f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8bf3088cc16f31c965831062ca57f3f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8bf3088cc16f31c965831062ca57f3f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8bf3088cc16f31c965831062ca57f3f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8bf3088cc16f31c965831062ca57f3f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a22e718904781f2a054d404d46db75c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f11c832fd0c754c3af36589dcd01cfb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10d4ecb5a2c0a23e5ce1d97e387d340c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26c496af661f594a452b0002b6269924
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 1, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_313bd69d0a61647b7968e06331a9f959(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_313bd69d0a61647b7968e06331a9f959(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_313bd69d0a61647b7968e06331a9f959(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_43bf3d180e1b4127fb8f910ec6cda8de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d5510158a522025748117912c9f3ad26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_518dacedf00c3e090ba7a1241a03047b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_518dacedf00c3e090ba7a1241a03047b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e55ffaab527eaed8f0f5b8221219e9e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.045360267162323], [0.0]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.12676018476486206], [0.01356402039527893], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_d96336a1dc7b5ba9b934546318ed8468(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.3706147074699402], [-0.4487587809562683], [-0.12151667475700378], [0.3006930649280548], [0.022215381264686584]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.3369906544685364], [0.01356402039527893], [-0.08927661180496216], [-0.22124694287776947], [-0.03989097476005554]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_77f5b5fd7ff228c93b780a88d7859677(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.009512990713119507], [-0.11628752946853638], [0.14816853404045105], [0.08443909883499146], [0.026305489242076874]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.12676018476486206], [0.24751397967338562], [0.01315334439277649], [-0.07173877209424973], [0.15080812573432922]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_67e209193749788b4150c50a904f2e4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.009512990713119507], [-0.095703125], [0.14816853404045105], [0.33977189660072327], [0.12315905839204788]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.3369906544685364], [0.24751397967338562], [0.09807157516479492], [0.06243331730365753], [0.19928254187107086]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_cc7f0b4fb76f9d48248a2a1152a450ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 68, 68], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e18583f65957528e14310ba9c5df1945(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_96e714155945114ff1a7f577b7ad3aa8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4f1387dd29e6414fd3ff99c81fb846c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_44369c558f56569fa5a14473e68165c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8bdf773e94530a41ad4b8dcb4ca5a711(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01421f7dff29dbb58036232da4159fa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e98a4f3b5e3959ecdeb00c7b75f915b
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_3605aca3e677045f208df2206e91238b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e98a4f3b5e3959ecdeb00c7b75f915b
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_0aaf2290134d34f1cb502ed2c21714d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e98a4f3b5e3959ecdeb00c7b75f915b
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_879278357d980c22197d2a1a86b4e6d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e98a4f3b5e3959ecdeb00c7b75f915b
    def get_inputs(self):
        return [
            paddle.to_tensor(16, dtype='int32').reshape([]),
            paddle.to_tensor(16, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_6fb1622b9c699d65e2ff5f2bca5df7e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e98a4f3b5e3959ecdeb00c7b75f915b
    def get_inputs(self):
        return [
            paddle.to_tensor(8, dtype='int32').reshape([]),
            paddle.to_tensor(8, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_714bccaf9368d67db27d6d4abbeb0ed8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 76, 116], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.27944546937942505], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cd35304399d08ce8a57dfaa1747960e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3b10475a6e6cc0c52542e48cca9654ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ec0f6fbf52fc756860697b827776a914(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_385be6c599090bb7ae0427bfb6ecaeb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1a77cfdb7c3bafd9e74ad702aaef9bb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e50269b1815b9963a82b2770b0e67980(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7cbf9ea34c096a043368dc76a2781b7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.30973464250564575], dtype='float32').reshape([1]),
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92efc39ee632ca2dacbad35e01f93870(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92efc39ee632ca2dacbad35e01f93870(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92efc39ee632ca2dacbad35e01f93870(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f3a8aef6a697056a492945ed6829c454(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_97bfb08daee242173288f3e5cf8e95f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 1248, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1248, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4fca9b0aa25d26407418712e2670c89d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bad56581bbfbcca0201668112489f610(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.143853098154068], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bad56581bbfbcca0201668112489f610(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.143853098154068], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_eb412a67c48ac57e19c6a6381a4a0742(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db3b9f26ac32814333fa349d6a1b5f77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.15872637927532196], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5868663be13fae4fb58dbb4084b94ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5868663be13fae4fb58dbb4084b94ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5868663be13fae4fb58dbb4084b94ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c0832b56a5e8930ecc95b64a0cc5b081(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([171, 480, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([171, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4a2605b3435a2b4b649f4a5cfc03b38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4a2605b3435a2b4b649f4a5cfc03b38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4a2605b3435a2b4b649f4a5cfc03b38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_011b4e42f34d6960d873d5e7a8f5d293(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_011b4e42f34d6960d873d5e7a8f5d293(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_011b4e42f34d6960d873d5e7a8f5d293(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0d780c39674e4c647f7bcc0d8c803398(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([145, 36, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([145, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e9741420440e75e62d7c52a6dec3537(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fda2a66d0faea7ebbb2ba163f4bfaccf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4203449785709381], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fc3aa377e8585c3b306b0006749aa8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c2d829508982377ced5f6ca7adf8a741(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e0521007cba695e6672ad861c616e36c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 23, 23], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_34abcf392d3d88263c0301a71cc77f84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_34abcf392d3d88263c0301a71cc77f84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_34abcf392d3d88263c0301a71cc77f84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_34abcf392d3d88263c0301a71cc77f84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_34abcf392d3d88263c0301a71cc77f84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_94a690dc1701a06bc9fa7cc9c24c1d45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2347, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2347, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_94a690dc1701a06bc9fa7cc9c24c1d45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2347, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2347, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_34abcf392d3d88263c0301a71cc77f84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4253eeb1adb0ca9e01ce6fd2c451457d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_97a9fc79a75b4ab73fb7c581116e975a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e61700a4c00b74ccbbd73e8222169592(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e61700a4c00b74ccbbd73e8222169592(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e61700a4c00b74ccbbd73e8222169592(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e61700a4c00b74ccbbd73e8222169592(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e61700a4c00b74ccbbd73e8222169592(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0d95e87a4b6bfe89ed4a071ae8cd33d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3109, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3109, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0d95e87a4b6bfe89ed4a071ae8cd33d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3109, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3109, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e61700a4c00b74ccbbd73e8222169592(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a1fa0b3a7afd9919d40c8d4183df2891(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a1fa0b3a7afd9919d40c8d4183df2891(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a1fa0b3a7afd9919d40c8d4183df2891(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a1fa0b3a7afd9919d40c8d4183df2891(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a1fa0b3a7afd9919d40c8d4183df2891(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8bb560dc6de25b4b9d620153d1746a44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3813, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3813, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8bb560dc6de25b4b9d620153d1746a44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3813, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3813, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a1fa0b3a7afd9919d40c8d4183df2891(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3ab706d0b5155fb8f5d698e0004f9fbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3886948227882385], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3ab706d0b5155fb8f5d698e0004f9fbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3886948227882385], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_73f4358d0262c75bb3ee116f27102ab7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9d97aca26459a8f32ab83f34f3b98f66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4644785523414612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4c083ac75710c5341d12c22b1aa6bc7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 156, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 156, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d75d43acfe44a663404af924d91265c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.9237934350967407]], [[0.8839898705482483]], [[0.8329225778579712]], [[0.9689385890960693]], [[0.9733434915542603]], [[0.8450244069099426]], [[0.9685894250869751]], [[0.944530189037323]], [[0.9532930254936218]], [[0.9226920008659363]], [[0.9564902782440186]], [[0.9280466437339783]], [[0.8986982107162476]], [[0.9537007808685303]], [[0.8488045334815979]], [[0.9264301061630249]], [[0.9475281834602356]], [[0.946185827255249]], [[0.9461697936058044]], [[0.9409826397895813]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_9912a4e350a7fec89bd5974abe852560(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d736be28db3d5162b6eff89fc799001(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a5c7de53ee11f6353c64ae5336f6558(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_56f399fd6b5f4097bc29fd5377049ac7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 92, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_69a58e50ba723730f651fa8a74868a8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c7a083d341c7f172f131c7c2e3178705(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a71a56db51d09b0d3ebaf31fa6951109(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ce94eab42c9529df535212c05d63ea08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_acf4e6732fc756199777a963ca110cb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 80, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_f77b09d6ec6db54f658968c74383ac82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 9, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f2418ad7e7ede2ae716ce43afa2d876f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f2418ad7e7ede2ae716ce43afa2d876f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f2418ad7e7ede2ae716ce43afa2d876f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e8fb9ed9cb28ba187e038a948f4e7c52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7cbf9ea34c096a043368dc76a2781b7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2748326361179352], dtype='float32').reshape([1]),
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f7d56b3427264e1a55bfa7e3e23c28ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d29d0875ad495c77f9439e92fe86e5ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5089c2a9eea05aba79fd5dc38f5b121f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 34, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bbabeebb9bc032b8337aed994556ef42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c341af0e76cde65225ec5a5c688492b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c49d37c4d6d0a3d2b6dd64ef79e011f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 872, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 872, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a50aa2dd342514e148d910d7c7377100(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([247], dtype='float32', min=0, max=0.5),
            paddle.uniform([247], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9c4968d0107ae2fd4af94b5ee119b59e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4f68a887c6f903395657dd4deddd22eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26c496af661f594a452b0002b6269924
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 1, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01bf15553e87aae7e1a88e7a7579119f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([22, 480, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_314110e5ca35c2f9cffdc692fdce893a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a1835a9564ecd6ddd929f02e9fca0167(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([145, 480, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([145, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7804a6b8089118134af99a3f8b44b158(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 40, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.49292048811912537], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9401a3ec65b218c2ec5b9635d6b05eb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([2, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.32936179637908936]], [[0.08513082563877106]]], dtype='float32').reshape([2, 1, 1]),
        ]


class TestPrimitiveOp_3c80d01c5167ba60864ed0bf86ec43bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 46, 46], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4d6d28ea0907b963441030ef5afbe21a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([171, 36, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([171, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_09eca576c63d1a0402716efd3485ddd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26c496af661f594a452b0002b6269924
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 1, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_812621aebda9b1628e231aa97758e715(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_102449b227531c8a349df5767e7844af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10284fefd5b41697a6365881e9675966(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 68, 68], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ca497507bbae5b5e0d044b2eef4321e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_95156cf7263983e071c2b73389b84cb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4c00802232d357dd8798f040abbac529(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26c496af661f594a452b0002b6269924
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 1, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7768be611e5fb093cc6b066614e2ab88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d9410d354f570de30c986dab7c6b4ce1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([2.0561161041259766, 1.9748029708862305, 1.9920729398727417, 2.24149227142334, 1.8799949884414673, 2.0826950073242188, 2.148406505584717, 2.2264139652252197, 2.2733545303344727, 2.1059107780456543, 1.964715600013733, 1.9450106620788574, 1.9700678586959839, 2.1986591815948486, 2.1880838871002197, 2.315531015396118, 1.9749348163604736, 2.1108779907226562, 2.026007652282715, 2.2071290016174316], dtype='float32').reshape([20]),
            paddle.to_tensor([0.5663484334945679, 0.851806104183197, 0.6317678093910217, 0.8800439834594727, 0.7044894695281982, 0.8345401287078857, 0.7639938592910767, 0.6662639379501343, 0.610576868057251, 0.7375533580780029, 0.6695274114608765, 0.9531345963478088, 0.8233892321586609, 0.835054337978363, 0.6011045575141907, 0.9829161763191223, 0.9560813903808594, 0.6947021484375, 0.8190240859985352, 0.6832518577575684], dtype='float32').reshape([20]),
        ]


class TestPrimitiveOp_b64a39a212d706809070e178f603c62a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([2.0416407585144043, 1.9545707702636719, 1.978739857673645, 2.056077241897583, 1.8828914165496826, 2.2567386627197266, 1.8941136598587036, 2.1774561405181885, 2.1005589962005615, 2.2876734733581543, 2.2010622024536133, 2.141038179397583, 2.168912887573242, 2.118363857269287, 2.162059783935547, 2.0070929527282715, 1.810794472694397, 2.152836322784424, 2.0113651752471924, 2.115436315536499], dtype='float32').reshape([20]),
            paddle.to_tensor([0.43365156650543213, 0.14819391071796417, 0.36823219060897827, 0.11995603144168854, 0.29551056027412415, 0.16545988619327545, 0.23600615561008453, 0.33373603224754333, 0.3894231617450714, 0.2624466121196747, 0.3304726183414459, 0.04686542972922325, 0.17661075294017792, 0.16494564712047577, 0.3988954424858093, 0.017083849757909775, 0.04391861706972122, 0.3052978515625, 0.18097592890262604, 0.31674817204475403], dtype='float32').reshape([20]),
        ]


class TestPrimitiveOp_81e299890c67c7c99272269dade39934(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5124597549438477, 0.4929511547088623, 0.4967908263206482, 0.5548126697540283, 0.47021275758743286, 0.5278730392456055, 0.5220979452133179, 0.5525187253952026, 0.5515159964561462, 0.5384034514427185, 0.5107054710388184, 0.4885494112968445, 0.5012965202331543, 0.5463536977767944, 0.5444257259368896, 0.5775654315948486, 0.4919314980506897, 0.5309219360351562, 0.505839467048645, 0.5445213913917542], dtype='float32').reshape([20]),
            paddle.to_tensor([0.3882734775543213, 0.017977707087993622, 0.1543045938014984, 0.2582632601261139, 0.3731561005115509, 0.26986953616142273, 0.2822590172290802, 0.16058045625686646, 0.269549697637558, 0.023253532126545906, 0.3000372052192688, 0.3573911488056183, 0.1051688864827156, 0.19059394299983978, 0.30486151576042175, 0.4724150002002716, 0.1532040238380432, 0.35600966215133667, 0.4785275459289551, 0.3040330410003662], dtype='float32').reshape([20]),
        ]


class TestPrimitiveOp_8b679e100eb93acdd107f5a17a39e33d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a6c37ad32745fc2dbdfcc189fc67183(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83615c216c9ac7f3bc83966db3aa4d22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b20ff19495a209f01f55207d4bbf65e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([247, 81], dtype='float32', min=0, max=0.5),
            paddle.uniform([247, 81], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_11fd0d976dc003c4302a241fc2602243(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_321f06501d93b36fe62f2264181ea9a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 84, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_82ee2e2b6b274ef60bc497fe4e1b483d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e921337b8323c73ee9eb9a411879473(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.923328161239624]], [[0.886096179485321]], [[0.8327062726020813]], [[0.9120669364929199]], [[0.8490750789642334]], [[0.8549423217773438]], [[0.7474304437637329]], [[0.9218114018440247]], [[0.8838335275650024]], [[0.8198943734169006]], [[0.948270857334137]], [[0.946869969367981]], [[0.8901512026786804]], [[0.8110017776489258]], [[0.9211472272872925]], [[0.8828393816947937]], [[0.9530349373817444]], [[0.9241278767585754]], [[0.8242674469947815]], [[0.9397584199905396]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_9912a4e350a7fec89bd5974abe852560(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d736be28db3d5162b6eff89fc799001(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d29d0875ad495c77f9439e92fe86e5ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0aaf2290134d34f1cb502ed2c21714d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e98a4f3b5e3959ecdeb00c7b75f915b
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_3605aca3e677045f208df2206e91238b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e98a4f3b5e3959ecdeb00c7b75f915b
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_01421f7dff29dbb58036232da4159fa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e98a4f3b5e3959ecdeb00c7b75f915b
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_829378df5108857c9137f56a405d8bb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_796c9e4a354976a224ebb08c28f47d67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bbabeebb9bc032b8337aed994556ef42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b7bc653b599b803c8a91128ea43b25ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f4eee5533aaa16e1147bad77d5b735b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f2ac02dd41348a1a2b277b460c4dc7fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_abcb88d508e4061dda859086121f3a24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4f68a887c6f903395657dd4deddd22eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26c496af661f594a452b0002b6269924
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 1, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5e0cf60edbcdf6c5d4eb522b82340ba1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6a786bd6f05ebbda9aaf4085b6865c47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3fcc108deeb9743466a6b9a43a5cfcc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_398d2a0f19e4364d0cba50614731d112(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cfc9f6a1dde4794c1eac28b4ac2ad84b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a855f0f09c04129ebe93db0280f30b9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([950, 81], dtype='float32', min=0, max=0.5),
            paddle.uniform([950, 81], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f4ba497b940916d09998a9094c70a840(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4025beb3242faa51d8253e81637bcbb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.0], [0.010479211807250977], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_5ac4b7d5551ed6e06fc197889081e6fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.01497313380241394], [-0.06235694885253906], [-0.15490691363811493], [-0.26111334562301636]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.16539761424064636], [0.40875089168548584], [0.022683806717395782], [0.30848658084869385]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_28e5de2f6d8df97e032cdaee689c8af8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.11871663480997086], [-0.1738341897726059], [-0.04759081453084946], [-0.09873183071613312]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.11057433485984802], [0.010479211807250977], [-0.20009133219718933], [-0.06530912220478058]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_a59042c46a25a294314e77d112be0c10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.11871663480997086], [0.01666581630706787], [0.11533603072166443], [-0.09873183071613312]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.0629366934299469], [0.40875089168548584], [0.1887095421552658], [0.30848658084869385]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_ab721ed89e519d9083ac542c2f38b807(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d6c318da2e143be2f423a053389bf091(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([70], dtype='float32', min=0, max=0.5),
            paddle.uniform([70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9282ea0696621093bf67c290e6509ab4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 480, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a39ffd2a100decc961d33f06312bc8f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f4ba497b940916d09998a9094c70a840(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b04c545f2969de86d56cf5a75f101276(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 76, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_666aa4b4a589bc4f519ac5831a8347c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 40, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4f3ceec01509c25b56e3a8c0cf1573ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1386e046812ece2f2dcf8ec49b631653(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e64dfaa412f663f2b547a8e7a33bee8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae515e81742ef660debb57062a5b3dd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae515e81742ef660debb57062a5b3dd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae515e81742ef660debb57062a5b3dd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae515e81742ef660debb57062a5b3dd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae515e81742ef660debb57062a5b3dd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f0532210c7fc18844134d0f7fd9d46eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2100, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2100, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f0532210c7fc18844134d0f7fd9d46eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2100, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2100, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae515e81742ef660debb57062a5b3dd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99eee3b0654cdd42bfa4fdc3800eb49c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_450a6f25943da53576ae6745383e6e25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f1c56ff192010c374b9eb05f64904b0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 21, 21], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_45f4999593fa4f0f5057b30da6575ecd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d9fb295dbb61d4dd71b5c60335d7410a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eed6a1c862589398a3b85184b26f7783(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([70, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([70, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_98d33fdd3f730babc8015e26f3fc5f82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_726bb629b41fcf4fb57aac755fa5f3c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7cbf9ea34c096a043368dc76a2781b7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.40933364629745483], dtype='float32').reshape([1]),
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5341e25811bf3e3921ae70bd937e2f87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_be126399eaf35d3626069f35eaed198d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26c496af661f594a452b0002b6269924
    def get_inputs(self):
        return [
            paddle.uniform([22, 2, 1, 9, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_85fc5d79194ca196e6bbdf261142378f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.14811408519744873], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_85fc5d79194ca196e6bbdf261142378f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.14811408519744873], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4f037ed61ff2b413fdd81115873eec10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_195269f23af787ccc75ba78276afe88c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3480321764945984], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fc28e52fc034806d4fe88c63e4b8f95d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d66ef87fdb80ed103f70a80347e7a58a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 9, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bbcd2eff46adb61e61f02434c322166f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 68, 68], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d409b9b7e56c506f1f394e82ed802f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([551], dtype='float32', min=0, max=0.5),
            paddle.uniform([551], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d6da5957b36ee63cbdfae0f73863728(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b38be7def7b30fc0725b5bfc5ae8248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_011b4e42f34d6960d873d5e7a8f5d293(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_011b4e42f34d6960d873d5e7a8f5d293(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_011b4e42f34d6960d873d5e7a8f5d293(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_011b4e42f34d6960d873d5e7a8f5d293(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10d4ecb5a2c0a23e5ce1d97e387d340c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26c496af661f594a452b0002b6269924
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 1, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eda12cb26dc3bf6b0f62af250b65f075(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_290eeb805aa7ea9fc539822d9e39bcea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c33c745db6018b960a4c702e4d32ad9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_95156cf7263983e071c2b73389b84cb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7a8d59909ba35e735c5601d021027f1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([3800], dtype='float32', min=0, max=0.5),
            paddle.uniform([3800], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ead8547a0bdcdd08caced02fd636d11f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fb0465d663fbc54d4952049367ea58ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_43589d2de714e21afb03bbab62fa4a8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 88, 88], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_da167c0b81f7842908f3e8575b04aa33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92efc39ee632ca2dacbad35e01f93870(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f2b7f336d884739eceb9638d2fa05d3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a83e91f4f960e08aa8e390d616556710(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 38, 58], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17097848653793335], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f3ee434ba12731eef495d58d467e7f30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4c74785e91eed091c5bd0bff16eeb031(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([2204], dtype='float32', min=0, max=0.5),
            paddle.uniform([2204], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d946dbd7af66e012fc5d3b1653d426da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4045492708683014], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3248be742df4d565e4d204e9ab9b0f2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ac04b8a4563d644e2992cc9c91a23b19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_29f00098c1714dbb564378c6a51cac97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_534c24bd1fc16e671936a7d705bb2fa1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.26069843769073486], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_97a9fc79a75b4ab73fb7c581116e975a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_09edd71a31f4e19882ae453a86730106(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae180e04ef411d9166814961f037b343(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae180e04ef411d9166814961f037b343(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae180e04ef411d9166814961f037b343(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae180e04ef411d9166814961f037b343(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3fcc108deeb9743466a6b9a43a5cfcc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2bd0537de3057dc5a8b6d5550e349b6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 23, 41], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2202b8d52ce4338d03964a3e9bb0df67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 46, 82], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_88e6775e311a77a14a466c9d02e0408c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 92, 164], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dafa079fcebcfc8735f8706d43af9185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c478ec56b33f95249fb5e4abcb0b9e47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 23, 41], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_117ba39f47ac37d10464705499b1d67f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 46, 82], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_a7b8bd77bba6cee3eeb0e5d45cc2946d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 92, 164], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_363443a50ae11b71092c2e7b8a9c5119(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_2d5ae1dd257ed56047c6b3aba8d4790f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 11, 17], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3413586914539337], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f65cc419fecb60ed7b0f1019cca1a044(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 17, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_233c3234c19fff2ea2fe6566505b37b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8a5bda07c8c5d72e5eb320ff277e286f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eb3e165cf46b4cbbcca4cfa7c3635ea5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.09021379053592682], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ffac11153f95203c006a1a0c75742043(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8938de35115b378936e3fdccd0de463d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8938de35115b378936e3fdccd0de463d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8938de35115b378936e3fdccd0de463d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_54a0b7d9eadf8273b83cf9cc989efc91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_54a0b7d9eadf8273b83cf9cc989efc91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_54a0b7d9eadf8273b83cf9cc989efc91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_54a0b7d9eadf8273b83cf9cc989efc91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_54a0b7d9eadf8273b83cf9cc989efc91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3c4ff594feb797f1a51a71143c4de8ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([4231, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4231, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3c4ff594feb797f1a51a71143c4de8ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([4231, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4231, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_54a0b7d9eadf8273b83cf9cc989efc91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0aaf2290134d34f1cb502ed2c21714d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e98a4f3b5e3959ecdeb00c7b75f915b
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_3605aca3e677045f208df2206e91238b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e98a4f3b5e3959ecdeb00c7b75f915b
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_01421f7dff29dbb58036232da4159fa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e98a4f3b5e3959ecdeb00c7b75f915b
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_6d7dfc6b0223d4d85fd0a6b9ab8cb2dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 19, 29], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.00022313621593639255], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5bda327c7fb5ca99999f82f1ea71320a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 624, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 624, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ec0b68923ab84191702ffc6976b9bde3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8938de35115b378936e3fdccd0de463d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8938de35115b378936e3fdccd0de463d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8938de35115b378936e3fdccd0de463d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8938de35115b378936e3fdccd0de463d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a5df384944d048c973b061054bb2b8fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_09eca576c63d1a0402716efd3485ddd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26c496af661f594a452b0002b6269924
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 1, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a918ea2a021c861f844ae2ecf5a1e976(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_09aa3bead01d7c78ba91ff04f54cddd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2f90baf999917130b1ee818f16971c
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.06484074890613556], dtype='float32').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()