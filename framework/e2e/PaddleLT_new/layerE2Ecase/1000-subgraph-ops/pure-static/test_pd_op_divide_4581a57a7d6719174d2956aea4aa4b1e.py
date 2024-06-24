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



class PrimitiveOp_1c7a4be3a8fbfed1cba7577260102b48(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 23, 23, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 23, 23, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6f71efd4914539c7ddf525656ca80079(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c7a4be3a8fbfed1cba7577260102b48
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6d32915e6cdc772607e041514b8f38a8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 11, 11, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 11, 11, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4b191f6fe67bd72fac9a5d9702c23286(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d32915e6cdc772607e041514b8f38a8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0bb22f30640d4643b4dd4415d6347ffe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 2100], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0cb6a7a943402794a6e1381bf5970ac9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bb22f30640d4643b4dd4415d6347ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.24377138912677765]]], dtype='float32').reshape([1, 1, 1]),
        ]


class PrimitiveOp_16d8bd42f7eff78b9e3230066b7a7608(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[12096, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[12096, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_54db2d2e7dc04d7320df17202705f995(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16d8bd42f7eff78b9e3230066b7a7608
    def get_inputs(self):
        return [
            paddle.uniform([12096, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([12096, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9288920304028b1ae986841ae437f48b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 24, 24, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 24, 24, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_de5e9df959118baa938638657d0092f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9288920304028b1ae986841ae437f48b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ef3619de02b323bad3de8a88a2380e0d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 42, 42, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 42, 42, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b89a3585efa1f06abe1a7f59d8c0e196(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef3619de02b323bad3de8a88a2380e0d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7dc936e5c979ff7ec0d7be99b1868efb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6832ef306b4e89666f6dd227567391bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7dc936e5c979ff7ec0d7be99b1868efb
    def get_inputs(self):
        return [
            paddle.to_tensor([1094.625732421875], dtype='float32').reshape([1]),
            paddle.to_tensor(8732.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_61a805987e1e8cb6a172636260758ae4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 21824], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 6, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c32cad66f8057e966def6b2261e16b45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61a805987e1e8cb6a172636260758ae4
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.0016840801108628511], [0.002114468952640891], [0.004909759387373924], [0.0025471397675573826], [0.003643445670604706], [0.0018383796559646726]]], dtype='float32').reshape([1, 6, 1]),
        ]


class TestPrimitiveOp_4b165e0684b47676f00e6e702da0968c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61a805987e1e8cb6a172636260758ae4
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.0014928891323506832], [0.0062005603685975075], [0.004184899386018515], [0.002309308620169759], [0.004033701494336128], [0.0033671867568045855]]], dtype='float32').reshape([1, 6, 1]),
        ]


class TestPrimitiveOp_0a714c45ba08bb22e87f3e03a0dfce2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61a805987e1e8cb6a172636260758ae4
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.147658109664917], [0.1335940808057785], [0.12027014046907425], [0.013102437369525433], [0.09767962247133255], [0.028695477172732353]]], dtype='float32').reshape([1, 6, 1]),
        ]


class PrimitiveOp_1aaaa64cbdaef9d69fa69ac5497e7945(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 46, 46, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 46, 46, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eea8e572135b6b14077dc10dec0d8161(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1aaaa64cbdaef9d69fa69ac5497e7945
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b1f39f603d43ab4921d299a6df37b647(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 12, 12, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 12, 12, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1a1174db1212b449e66246d3466965fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1f39f603d43ab4921d299a6df37b647
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6f71efd4914539c7ddf525656ca80079(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c7a4be3a8fbfed1cba7577260102b48
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_42844b690326291e4cdd8ea95897c54d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 84, 84, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 84, 84, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f4ddbe3bae510309103390cd74d98b4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42844b690326291e4cdd8ea95897c54d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d9b6439532c9ba79072152eb89e60f87(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7b7277faccf5e0b271b5d449a96b5415(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(8.13055419921875, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_45f4c2efa1722b84fb1c18b26f9efa3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(2.4867758750915527, dtype='float32').reshape([]),
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7045764ff1606a1c2114ef7d98ab1c2c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 38, 38, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 38, 38, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2c135ae4e01b8d93b1a2b10c93269c0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7045764ff1606a1c2114ef7d98ab1c2c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7701b6190833ed4ce60c44eb82719da8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1786, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1786, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7af3f44230d8ca479b1cffc3d0a410ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7701b6190833ed4ce60c44eb82719da8
    def get_inputs(self):
        return [
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7af3f44230d8ca479b1cffc3d0a410ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7701b6190833ed4ce60c44eb82719da8
    def get_inputs(self):
        return [
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_79e435cfb2075fe7db9eecdd14c7a392(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(-918214.75, dtype='float32').reshape([]),
            paddle.to_tensor([0.18549604713916779], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b0d133dc9f9b69562ecf4d3ef6f1ebb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(101438.5859375, dtype='float32').reshape([]),
            paddle.to_tensor([0.18549604713916779], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a86973dd1105d6dc47ad96d533066ad2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(963.9410400390625, dtype='float32').reshape([]),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c0ae64e72e5aa4e75259ac44863cd3ca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 48, 48, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 48, 48, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0b54c003e4f5f9fa067367e36ae0900c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0ae64e72e5aa4e75259ac44863cd3ca
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_13ef74aa9f50b42fe3d36a1229797140(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 21, 21, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 21, 21, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2212ad03b5dca1b4a55949506d514108(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13ef74aa9f50b42fe3d36a1229797140
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_345d2d085813649aee260a0dfb3ff397(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5376, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[5376, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ab9ff37f33666ce96947a55c3c9dee12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_345d2d085813649aee260a0dfb3ff397
    def get_inputs(self):
        return [
            paddle.uniform([5376, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([5376, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_94a67a6b88168f05f1fe8eb9f0eb46ca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 44, 44, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 44, 44, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_45ddea591d90d43aa94e5aaa38f4bf75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94a67a6b88168f05f1fe8eb9f0eb46ca
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_849ec7c645ad4a347bd586ebdbc99fcc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 92, 92, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 92, 92, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_34527f69743057bf8ab09e0762ac07d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_849ec7c645ad4a347bd586ebdbc99fcc
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d6d673aa53487facfa88844d66a76af7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[9, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[9, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dd2e764f55542298ce0ad1e54485642a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6d673aa53487facfa88844d66a76af7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.012567024677991867], [0.033545464277267456], [-0.07613429427146912], [-0.005593098234385252], [-0.0885133370757103], [0.10474269092082977], [-0.025808751583099365], [0.003745029680430889], [0.005792463663965464]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_abb483f2d595ad574e4726f878ad608d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6d673aa53487facfa88844d66a76af7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.14666686952114105], [0.08305582404136658], [0.07497528195381165], [0.02311849407851696], [0.11179906874895096], [-0.0416865274310112], [0.12510141730308533], [-0.0013537590857595205], [0.0003203027881681919]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.15923389792442322], [0.11660128831863403], [-0.0011590110370889306], [0.01752539537847042], [0.02328573353588581], [0.06305616348981857], [0.09929267317056656], [0.0023912705946713686], [0.0061127664521336555]], dtype='float32').reshape([9, 1]),
        ]


class PrimitiveOp_51de3ec31ab1ea3848e4c265af096298(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 8, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_790724226f42e117b81d24c8b5ae3a77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51de3ec31ab1ea3848e4c265af096298
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3735694885253906], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_82193780171fa4e34617eee4aaec1f35(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5529, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[5529, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3a14e73ad3ab64ef17ea1eca3b9cd1a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_82193780171fa4e34617eee4aaec1f35
    def get_inputs(self):
        return [
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3a14e73ad3ab64ef17ea1eca3b9cd1a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_82193780171fa4e34617eee4aaec1f35
    def get_inputs(self):
        return [
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bad5fcc14f08f7e5511445e901145e43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(10183.986328125, dtype='float32').reshape([]),
            paddle.to_tensor([0.4190918505191803], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b8ef268d83e15d912fbf436dd18824c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(3951.8916015625, dtype='float32').reshape([]),
            paddle.to_tensor([0.4190918505191803], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_e39c526fd111fcd8d7384d75c9d26791(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6], dtype='float32'),
            paddle.static.InputSpec(shape=[6], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0f6264615e28c330ea9065708bd3a08f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e39c526fd111fcd8d7384d75c9d26791
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0, -0.0, -0.0], dtype='float32').reshape([6]),
            paddle.to_tensor([0.01054816972464323, 0.011631745845079422, -0.09121815115213394, -0.0017405537655577064, 0.04042661190032959, -0.021534491330385208], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_2536b9ed81757070fd577b5b7e1d1bb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e39c526fd111fcd8d7384d75c9d26791
    def get_inputs(self):
        return [
            paddle.to_tensor([0.05690630525350571, 0.0607171431183815, 0.10605637729167938, 0.10256434231996536, 0.08725559711456299, 0.10694298893213272], dtype='float32').reshape([6]),
            paddle.to_tensor([0.19252170622348785, 0.17006026208400726, 0.28895995020866394, 0.13612869381904602, 0.311922550201416, 0.1498410999774933], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_37463832d340e1b90bea320a5f2f0f19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e39c526fd111fcd8d7384d75c9d26791
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.17463308572769165, -0.2213263213634491, -0.3390448987483978, -0.017620444297790527, 0.368111252784729, 0.18772169947624207], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.006831318140029907, -0.05255473405122757, 0.2690444588661194, 0.09878036379814148, 0.10982172191143036, -0.16934990882873535], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_45936156d11ae6c07c6ffea9a94649cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e39c526fd111fcd8d7384d75c9d26791
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3788352608680725, 0.11308163404464722, 0.2824884057044983, 0.2018137127161026, -0.006253153085708618, 0.09728133678436279], dtype='float32').reshape([6]),
            paddle.to_tensor([0.02469462901353836, -0.3824889659881592, -0.05592024326324463, -0.12581928074359894, 0.1565726101398468, 0.1054278314113617], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_cadfdbfa05683766ec180e21257b61b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e39c526fd111fcd8d7384d75c9d26791
    def get_inputs(self):
        return [
            paddle.to_tensor([0.00027387519367039204, 1.0703620910644531, 0.09158029407262802, 0.28379061818122864, 0.7070046663284302, 1.0143624544143677], dtype='float32').reshape([6]),
            paddle.to_tensor([1.0002738237380981, 2.070362091064453, 1.0915802717208862, 1.2837905883789062, 1.7070046663284302, 2.014362335205078], dtype='float32').reshape([6]),
        ]


class PrimitiveOp_adffc11cf87e8b93c302bd13fb881d4a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1767, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1767, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8874706a6a842bc030b51de8b17f8103(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adffc11cf87e8b93c302bd13fb881d4a
    def get_inputs(self):
        return [
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8874706a6a842bc030b51de8b17f8103(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adffc11cf87e8b93c302bd13fb881d4a
    def get_inputs(self):
        return [
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6fc05b66b95568981b09286e8a33eeb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(121523.828125, dtype='float32').reshape([]),
            paddle.to_tensor([0.28091827034950256], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_13f78d7b7cb9da5ce277b2911f71a204(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(104006.4375, dtype='float32').reshape([]),
            paddle.to_tensor([0.28091827034950256], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5c357742346c027717fa73d92402b7c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(948.8930053710938, dtype='float32').reshape([]),
            paddle.to_tensor([4.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_4943d5957299860748a46c4b99879007(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8400, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[8400, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5c4f36bb817eeb396d8999863494d089(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4943d5957299860748a46c4b99879007
    def get_inputs(self):
        return [
            paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([8400, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1b55e6c640098b1b5f55c96425d2b8ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 38, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1, 38, 38], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1491841c8ff5b160711d76015fddec50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b55e6c640098b1b5f55c96425d2b8ae
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 38, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b89a3585efa1f06abe1a7f59d8c0e196(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef3619de02b323bad3de8a88a2380e0d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2212ad03b5dca1b4a55949506d514108(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13ef74aa9f50b42fe3d36a1229797140
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_770cec0c14714f051bdf4166dc39f2ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1490, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1490, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f1694ccfee662bc32a04a1e70ceca4d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_770cec0c14714f051bdf4166dc39f2ef
    def get_inputs(self):
        return [
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f1694ccfee662bc32a04a1e70ceca4d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_770cec0c14714f051bdf4166dc39f2ef
    def get_inputs(self):
        return [
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1fa1553de0ad2fb18f63a30d00c2351a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(-220589.5625, dtype='float32').reshape([]),
            paddle.to_tensor([0.49846354126930237], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1f8c704c439ec6906275cce3690d4668(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(84985.5703125, dtype='float32').reshape([]),
            paddle.to_tensor([0.49846354126930237], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_f45f300d98a23fb80ce3b79163737f4c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 3549], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 2, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e0fadc2268f3ca53b016ba0f66dab3bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f45f300d98a23fb80ce3b79163737f4c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.24032799899578094], [0.2445775866508484]]], dtype='float32').reshape([1, 2, 1]),
        ]


class TestPrimitiveOp_1a1174db1212b449e66246d3466965fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1f39f603d43ab4921d299a6df37b647
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4b191f6fe67bd72fac9a5d9702c23286(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d32915e6cdc772607e041514b8f38a8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1b5503add412187f8e75c884e9eb3420(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2fac50ba8f97e1a479322f322d215b02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b5503add412187f8e75c884e9eb3420
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.02668238990008831]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_7f74840be6eb1d546674d5bf840f9617(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b5503add412187f8e75c884e9eb3420
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06038498878479004]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.0870673805475235]], dtype='float32').reshape([1, 1]),
        ]


class PrimitiveOp_8357bb4e351fddef99148117a3fc5989(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[6, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6497d3d1cde524ae11bf14bd4d55ee7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8357bb4e351fddef99148117a3fc5989
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0015165689401328564], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.0046247332356870174], [0.003697821171954274], [0.06876641511917114], [-0.03426813334226608], [0.043325275182724], [0.004558820743113756]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_a8f8e5c0e399fbd2819f3ea86151c572(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8357bb4e351fddef99148117a3fc5989
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0024645517114549875], [0.004094772040843964], [0.03775056451559067], [0.07193854451179504], [-0.03332417830824852], [-0.0037859706208109856]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.00216018152423203], [0.0077925934456288815], [0.10651697963476181], [0.03767040744423866], [0.010001097805798054], [0.0007728502387180924]], dtype='float32').reshape([6, 1]),
        ]


class PrimitiveOp_67d965451e79a3207d538c5bcb8f9776(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 4116], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_717afe3ae5bf3601190ce6b98807df30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67d965451e79a3207d538c5bcb8f9776
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.2459918111562729]]], dtype='float32').reshape([1, 1, 1]),
        ]


class PrimitiveOp_09f9c9507da7244963925ab3fcd8d64f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 19, 34], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0d60be30a2a0201327685c0ad9d7e04e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09f9c9507da7244963925ab3fcd8d64f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0b54c003e4f5f9fa067367e36ae0900c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0ae64e72e5aa4e75259ac44863cd3ca
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_de5e9df959118baa938638657d0092f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9288920304028b1ae986841ae437f48b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ede22d142ae7b21f573e1a8981915063(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(58.5085563659668, dtype='float32').reshape([]),
            paddle.to_tensor([7.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_378771470a1129010dd57683d1f37b4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(549.608154296875, dtype='float32').reshape([]),
            paddle.to_tensor([4.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7827d3f7553defab3ae3be8c20a4b77b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2010, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2010, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_be5811dc742556bdd7d58a21962afd72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7827d3f7553defab3ae3be8c20a4b77b
    def get_inputs(self):
        return [
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_be5811dc742556bdd7d58a21962afd72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7827d3f7553defab3ae3be8c20a4b77b
    def get_inputs(self):
        return [
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92c777f80d5228384d5860640a92d55c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(88068.125, dtype='float32').reshape([]),
            paddle.to_tensor([0.015555799938738346], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8c91e2d3673754960dc3eb91aa916653(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(114124.25, dtype='float32').reshape([]),
            paddle.to_tensor([0.015555799938738346], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_b96bf5cb1822fdef70540ddc601a781a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 152, 272], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_17d632ef29874c4d840f49b2cf3cf870(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b96bf5cb1822fdef70540ddc601a781a
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0928d1a760bb9d25cd90186c07cd625e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 22, 22, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 22, 22, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_239722ec4fd5e0a4956e6f88f328cb54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0928d1a760bb9d25cd90186c07cd625e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_477836b2efc66ca1e0c3afd9e02f7ad3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e3a37407fb54596f4bdd5159e66912f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_477836b2efc66ca1e0c3afd9e02f7ad3
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.14177025854587555], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_e3cc169f6196154b9395afcab5ad43eb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4663, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[4663, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f6e6360cd663a692f735b5ed512a4de6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3cc169f6196154b9395afcab5ad43eb
    def get_inputs(self):
        return [
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f6e6360cd663a692f735b5ed512a4de6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3cc169f6196154b9395afcab5ad43eb
    def get_inputs(self):
        return [
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c2c340ee1b29b344a14b238ce62a663a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(1132136.0, dtype='float32').reshape([]),
            paddle.to_tensor([0.12881667912006378], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2961324f05f63f57dda2dcf23bdd3490(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(265396.65625, dtype='float32').reshape([]),
            paddle.to_tensor([0.12881667912006378], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d86234b69c56c851a9440c6c8d1fa490(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7dc936e5c979ff7ec0d7be99b1868efb
    def get_inputs(self):
        return [
            paddle.to_tensor([304.9909362792969], dtype='float32').reshape([1]),
            paddle.to_tensor(2434.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_a85a5e8e53e1f8efd80ced8440076483(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1090, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1090, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8ccf7d0a58ddf89a85051f1d3232a29b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a85a5e8e53e1f8efd80ced8440076483
    def get_inputs(self):
        return [
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8ccf7d0a58ddf89a85051f1d3232a29b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a85a5e8e53e1f8efd80ced8440076483
    def get_inputs(self):
        return [
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_743dc98c53516659ff1ce9d7c1856bba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(17909.66015625, dtype='float32').reshape([]),
            paddle.to_tensor([0.3346521556377411], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e1d4134833d2263b3d99eb10d152fac3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(15460.2080078125, dtype='float32').reshape([]),
            paddle.to_tensor([0.3346521556377411], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_eea8e572135b6b14077dc10dec0d8161(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1aaaa64cbdaef9d69fa69ac5497e7945
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e7a7e07c98df6632d59175511f8874a8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[100, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9fa8743ffc0b78ae9ef010f511af9461(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7a7e07c98df6632d59175511f8874a8
    def get_inputs(self):
        return [
            paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_148d6ac01daccbf3ef5985d4a721ccdc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[2, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_374ea1cbbbaad2328a6ba865b4b3e2ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_148d6ac01daccbf3ef5985d4a721ccdc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.44256559014320374, 0.48185357451438904, 0.005073907319456339, 0.33582615852355957], [0.12907643616199493, 0.39162465929985046, 0.3457605838775635, 0.42338287830352783]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([[0.25629809498786926, 0.47895127534866333, 0.0592893548309803, 0.41566580533981323], [0.31651610136032104, 0.32637420296669006, 0.14966227114200592, 0.4332932233810425]], dtype='float32').reshape([2, 4]),
        ]


class TestPrimitiveOp_45ddea591d90d43aa94e5aaa38f4bf75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94a67a6b88168f05f1fe8eb9f0eb46ca
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7a621c5d797a61a537bd3d1833d32e41(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6069, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[6069, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f9938de2e816ed3abe153d044274abb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a621c5d797a61a537bd3d1833d32e41
    def get_inputs(self):
        return [
            paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([6069, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2ca4e5ab2c2ee85c9b5fa42e6272cb60(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[300, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_db855d2c6d0649f77a784c82648b2cce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ca4e5ab2c2ee85c9b5fa42e6272cb60
    def get_inputs(self):
        return [
            paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa28a1f33da18a62fddd08c548109479(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_148d6ac01daccbf3ef5985d4a721ccdc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.18591265380382538, 0.2508420944213867, 0.15098348259925842, 0.10198328644037247], [0.1829240769147873, 0.21212270855903625, 0.21822214126586914, 0.3630456030368805]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([[0.11228305846452713, 0.017179569229483604, 0.4269137382507324, 0.06641856580972672], [0.36689719557762146, 0.426728218793869, 0.3064291477203369, 0.4882761240005493]], dtype='float32').reshape([2, 4]),
        ]


class PrimitiveOp_7c84f2d13b6bd2bd2f44408312c90b78(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[5, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_40fe54f9d0307d2d129abce1cfa1cea3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c84f2d13b6bd2bd2f44408312c90b78
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.007825963199138641], [-0.07206472009420395], [0.007157676853239536], [-0.01321999728679657], [0.1348765641450882]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_1bc2f69daee6d6831ae16c9213a24e85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c84f2d13b6bd2bd2f44408312c90b78
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0010752808302640915], [0.10442597419023514], [-0.007352243643254042], [0.02358059771358967], [0.02060827612876892]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.008901244029402733], [0.03236125409603119], [-0.00019456679001450539], [0.010360600426793098], [0.15548484027385712]], dtype='float32').reshape([5, 1]),
        ]


class PrimitiveOp_39d9ef8c2243d1b1e1d36f72628e322b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 128, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_56a5f3b3f42d8ce12e531db661797323(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39d9ef8c2243d1b1e1d36f72628e322b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10970340669155121], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2c135ae4e01b8d93b1a2b10c93269c0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7045764ff1606a1c2114ef7d98ab1c2c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_34527f69743057bf8ab09e0762ac07d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_849ec7c645ad4a347bd586ebdbc99fcc
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d76b5ec1e1c4784a6ec33688d260f4c9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 19, 19, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 19, 19, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_70ed80ed92a2c736425a5b17535a7314(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d76b5ec1e1c4784a6ec33688d260f4c9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f8516f31ab1ce124214ec2130cd75196(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2374, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2374, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_acac861f6fc32cd7619812602aa75f01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8516f31ab1ce124214ec2130cd75196
    def get_inputs(self):
        return [
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_acac861f6fc32cd7619812602aa75f01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8516f31ab1ce124214ec2130cd75196
    def get_inputs(self):
        return [
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0d0ec158056f838d2e12eeb52a68e8bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(210676.625, dtype='float32').reshape([]),
            paddle.to_tensor([0.2393982708454132], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1bb0563c92ff1f6e74949d6bef491ed5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(134688.65625, dtype='float32').reshape([]),
            paddle.to_tensor([0.2393982708454132], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_2c6f0ed4d1ef56a5d95f1bc2dc9eff72(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3058, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[3058, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aa6a787d671b1714d9a208857d6b2486(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c6f0ed4d1ef56a5d95f1bc2dc9eff72
    def get_inputs(self):
        return [
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa6a787d671b1714d9a208857d6b2486(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c6f0ed4d1ef56a5d95f1bc2dc9eff72
    def get_inputs(self):
        return [
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3b947966e4cedf059b94d8512849fce2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(25487552.0, dtype='float32').reshape([]),
            paddle.to_tensor([0.2094583958387375], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_84534a711ff5d34fb865a27b8d7178c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(173918.75, dtype='float32').reshape([]),
            paddle.to_tensor([0.2094583958387375], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_b1772f1e2fcf5812207d6d05d95b522f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3793, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[3793, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_999be504ce09820b4059b23d47438745(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1772f1e2fcf5812207d6d05d95b522f
    def get_inputs(self):
        return [
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_999be504ce09820b4059b23d47438745(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1772f1e2fcf5812207d6d05d95b522f
    def get_inputs(self):
        return [
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4797183d53e1a52576ad246fd51689b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(3193316.0, dtype='float32').reshape([]),
            paddle.to_tensor([0.12741294503211975], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0aa814156472f51521bc68e8ba9d8716(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(215562.734375, dtype='float32').reshape([]),
            paddle.to_tensor([0.12741294503211975], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_b264268af78d24055b659edf41a403e5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b332e0f48baf22be3bf6c9dcf83ffe69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b264268af78d24055b659edf41a403e5
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.19838325679302216], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_239722ec4fd5e0a4956e6f88f328cb54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0928d1a760bb9d25cd90186c07cd625e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e0bf9923156201c4aed885e98a356f69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(15.385972023010254, dtype='float32').reshape([]),
            paddle.to_tensor([3.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_210b6b79c731e4aba93f558e9047993b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20267, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[20267, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d502cef41ffb864084ed36bfb5a2580d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_210b6b79c731e4aba93f558e9047993b
    def get_inputs(self):
        return [
            paddle.uniform([20267, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([20267, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_64508d0b7fb630d6b59223975ab7586e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_51e16a4d17903fbc13bbc85d0e32f48a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64508d0b7fb630d6b59223975ab7586e
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.053213175386190414], [0.057551003992557526], [0.008876675739884377], [-0.0954962968826294]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_c62237e89f501e19181f607be6c266cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64508d0b7fb630d6b59223975ab7586e
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.028628386557102203], [0.13692429661750793], [-0.021772535517811775], [0.11912152171134949]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.02458478882908821], [0.19447529315948486], [-0.012895859777927399], [0.023625224828720093]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_a4858dec6f9fa7189b8ba82014c7fd2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(3.7586605548858643, dtype='float32').reshape([]),
            paddle.to_tensor([7.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_83fb846fb60474cd2852d6f80b5c1a3d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2042, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2042, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_820d2a2b36d5cd8251de57e5df03abed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_83fb846fb60474cd2852d6f80b5c1a3d
    def get_inputs(self):
        return [
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_820d2a2b36d5cd8251de57e5df03abed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_83fb846fb60474cd2852d6f80b5c1a3d
    def get_inputs(self):
        return [
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_93bc8cec805e6d7c102c4676f3dd7f49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(-16014.9541015625, dtype='float32').reshape([]),
            paddle.to_tensor([0.2982756197452545], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_71362cf7949513e1a74fdaa8455ff5f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(29075.29296875, dtype='float32').reshape([]),
            paddle.to_tensor([0.2982756197452545], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_cb2c9b3aa1051c2cf862c2846a4f657b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_84e3bd7c35aaae7fde0500d59e4b814a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb2c9b3aa1051c2cf862c2846a4f657b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.36707356572151184], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_68f3ebabeb8c375ef03d86e1a1b36aa1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(35.623043060302734, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7e1f1271690e70375c15c3d7e621d6eb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6804, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[6804, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_69b93758a789ef7907850304ca99e29a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e1f1271690e70375c15c3d7e621d6eb
    def get_inputs(self):
        return [
            paddle.uniform([6804, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([6804, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_70ed80ed92a2c736425a5b17535a7314(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d76b5ec1e1c4784a6ec33688d260f4c9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6b8bb491a9c962d4203751002b15b84a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(235.47879028320312, dtype='float32').reshape([]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3e5a40e93a9c1858c7372558a7f50755(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(137.48477172851562, dtype='float32').reshape([]),
            paddle.to_tensor([7.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f4ddbe3bae510309103390cd74d98b4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42844b690326291e4cdd8ea95897c54d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ad6ecad809d350691b47c24d8a26d762(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 76, 76, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 76, 76, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dfba097a582f0e195a1dd127d9251ce5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad6ecad809d350691b47c24d8a26d762
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_37a46a10215784325f743582c1d36144(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4185, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[4185, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1bd00d24ece72a6de2a1ea6045723a07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37a46a10215784325f743582c1d36144
    def get_inputs(self):
        return [
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1bd00d24ece72a6de2a1ea6045723a07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37a46a10215784325f743582c1d36144
    def get_inputs(self):
        return [
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_479cdb16de69d75f39fc9c23aa20c028(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(-6850173.0, dtype='float32').reshape([]),
            paddle.to_tensor([0.20850594341754913], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f81f5ec601cf7392cb0deb6cb883e147(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(237587.5625, dtype='float32').reshape([]),
            paddle.to_tensor([0.20850594341754913], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_dfba097a582f0e195a1dd127d9251ce5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad6ecad809d350691b47c24d8a26d762
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()