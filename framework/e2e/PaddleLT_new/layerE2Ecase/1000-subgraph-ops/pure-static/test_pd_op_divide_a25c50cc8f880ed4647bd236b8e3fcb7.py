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


class TestPrimitiveOp_617171933004a8b3e855ef9d1211ae75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bb22f30640d4643b4dd4415d6347ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.2400422841310501]]], dtype='float32').reshape([1, 1, 1]),
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


class TestPrimitiveOp_8b15c2adebde27bdc5d0302962b03bf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7dc936e5c979ff7ec0d7be99b1868efb
    def get_inputs(self):
        return [
            paddle.to_tensor([1079.2420654296875], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_3566712ef8184630e6aea049aefe8970(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61a805987e1e8cb6a172636260758ae4
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.0008454338531009853], [0.051650192588567734], [0.00016203560517169535], [0.002629276365041733], [0.007275352720171213], [0.00992299523204565]]], dtype='float32').reshape([1, 6, 1]),
        ]


class TestPrimitiveOp_cb847abc9a888f79df6bf5e0d5a8a2c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61a805987e1e8cb6a172636260758ae4
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.01194661296904087], [0.058344822376966476], [0.00036035384982824326], [0.0010777389397844672], [0.0006172082503326237], [0.00040709454333409667]]], dtype='float32').reshape([1, 6, 1]),
        ]


class TestPrimitiveOp_051222c14640d99b26eb3daa7ee4ec45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61a805987e1e8cb6a172636260758ae4
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.08853987604379654], [0.17722313106060028], [0.010270973667502403], [0.03808582201600075], [0.05993069335818291], [0.09734964370727539]]], dtype='float32').reshape([1, 6, 1]),
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


class TestPrimitiveOp_5637d55becdb59a7ddefee5a10814af5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(10.433246612548828, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_384cb533f4a7ab84001e1456f27b6e00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(2.925077438354492, dtype='float32').reshape([]),
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


class PrimitiveOp_3fe4f165191a134fcb73fe1e8fe324d8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1758, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1758, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5869bba1ae3f37f347befea035177ece(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3fe4f165191a134fcb73fe1e8fe324d8
    def get_inputs(self):
        return [
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5869bba1ae3f37f347befea035177ece(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3fe4f165191a134fcb73fe1e8fe324d8
    def get_inputs(self):
        return [
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd043e03a35628b860e896295972ce5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(101207.6953125, dtype='float32').reshape([]),
            paddle.to_tensor([0.4881781339645386], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7303433935fd391e8dce3e20fafd6ddc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(99827.65625, dtype='float32').reshape([]),
            paddle.to_tensor([0.4881781339645386], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1f3f75a4dbf98082d08aba1429ab3ee7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(938.5811157226562, dtype='float32').reshape([]),
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


class TestPrimitiveOp_fd827ff414e7d55f67421aff339d550e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6d673aa53487facfa88844d66a76af7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.019558507949113846], [-0.06129153072834015], [0.08375721424818039], [0.0771404579281807], [-0.013100765645503998], [-0.012871161103248596], [0.017001457512378693], [-0.071235790848732], [-0.07514406740665436]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_81fb5aa11179adc99f3f4316ad9c8e2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6d673aa53487facfa88844d66a76af7
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.03754547983407974], [0.08032960444688797], [0.019050128757953644], [-0.07740714401006699], [0.04317725449800491], [-0.05373253673315048], [-0.002521224319934845], [0.07187458127737045], [0.07421426475048065]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.017986971884965897], [0.019038071855902672], [0.10280734300613403], [-0.000266688090050593], [0.030076488852500916], [-0.06660369783639908], [0.014480233192443848], [0.0006387874018400908], [-0.0009298031218349934]], dtype='float32').reshape([9, 1]),
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


class TestPrimitiveOp_81ab524d334a4652f763d15961d2d108(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51de3ec31ab1ea3848e4c265af096298
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4411681294441223], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_e16d749b5a6062042abdc1c0a9c9188d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5593, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[5593, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e9c97e41f7309a5d078bf1646bfda21f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e16d749b5a6062042abdc1c0a9c9188d
    def get_inputs(self):
        return [
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e9c97e41f7309a5d078bf1646bfda21f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e16d749b5a6062042abdc1c0a9c9188d
    def get_inputs(self):
        return [
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4b6738651c925b96b03db440066bb77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(563.8095703125, dtype='float32').reshape([]),
            paddle.to_tensor([0.19686909019947052], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ef01804b5584b1d4c73288e03639f845(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(3981.27685546875, dtype='float32').reshape([]),
            paddle.to_tensor([0.19686909019947052], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_ea7805aee9a7e5c03aec33f4c352c865(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e39c526fd111fcd8d7384d75c9d26791
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, -0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([6]),
            paddle.to_tensor([0.025464775040745735, 0.018943173810839653, -0.0040554567240178585, 0.004251727368682623, -0.0801314041018486, -0.034277696162462234], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_e97217aada549ab57dee32adf8b50b57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e39c526fd111fcd8d7384d75c9d26791
    def get_inputs(self):
        return [
            paddle.to_tensor([0.022082287818193436, 0.0017033217009156942, 0.016980469226837158, 0.09646375477313995, 0.04527043178677559, 0.04262499511241913], dtype='float32').reshape([6]),
            paddle.to_tensor([0.024973386898636818, 0.16535016894340515, 0.04132317006587982, 0.01826058328151703, 0.1689380407333374, 0.11583992838859558], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_9dcfdfcba910df14e20f3776b082fe3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e39c526fd111fcd8d7384d75c9d26791
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.09959647059440613, 0.05746108293533325, 0.024996191263198853, -0.01376459002494812, -0.2947706878185272, -0.12244720757007599], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.25567948818206787, 0.32966962456703186, -0.16224297881126404, -0.3088887929916382, 0.27184319496154785, 0.27993857860565186], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_a58cb83351076e603c553d6e97bd7168(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e39c526fd111fcd8d7384d75c9d26791
    def get_inputs(self):
        return [
            paddle.to_tensor([0.15271055698394775, -0.4032127261161804, 0.09368011355400085, -0.11237797141075134, -0.3011571168899536, -0.0737823098897934], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.053670018911361694, 0.06439828872680664, -0.3025127351284027, -0.037513989955186844, -0.1878737211227417, -0.1869288831949234], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_bcc9bfbf57db8e6663912fa50f95c08e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e39c526fd111fcd8d7384d75c9d26791
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0430938005447388, 1.018149733543396, 0.008810680359601974, 0.5875817537307739, 1.3704513311386108, 0.25183093547821045], dtype='float32').reshape([6]),
            paddle.to_tensor([2.043093681335449, 2.0181498527526855, 1.0088106393814087, 1.587581753730774, 2.3704514503479004, 1.2518309354782104], dtype='float32').reshape([6]),
        ]


class PrimitiveOp_ec9c56d606a0ddb3c7de9c03f3e1cbe3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1763, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1763, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7ca23ab65d4b9657ccc60846d99c1625(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec9c56d606a0ddb3c7de9c03f3e1cbe3
    def get_inputs(self):
        return [
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7ca23ab65d4b9657ccc60846d99c1625(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec9c56d606a0ddb3c7de9c03f3e1cbe3
    def get_inputs(self):
        return [
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_96158dab6b82a3d163c735f2d60828c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(147555.296875, dtype='float32').reshape([]),
            paddle.to_tensor([0.43052923679351807], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_00d85b150724bdf8c55d887f76e32cd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(104068.3828125, dtype='float32').reshape([]),
            paddle.to_tensor([0.43052923679351807], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_131cb4758eaa5ac88d88872df44888bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(944.32958984375, dtype='float32').reshape([]),
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


class TestPrimitiveOp_1f644c2bea90338afb5225e37562e200(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(44007.7734375, dtype='float32').reshape([]),
            paddle.to_tensor([0.0532023087143898], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_92ce96eae5d9b9cdab78d9da3350e874(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(84846.359375, dtype='float32').reshape([]),
            paddle.to_tensor([0.0532023087143898], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_abb0280415578eb4cc1c50c786bf5dae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f45f300d98a23fb80ce3b79163737f4c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.24741443991661072], [0.24345114827156067]]], dtype='float32').reshape([1, 2, 1]),
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


class TestPrimitiveOp_45a21e314139af9d5e6a74aa07156bdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b5503add412187f8e75c884e9eb3420
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.0033264346420764923]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_c75bed9a4ec2abdece608788a8b14f5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b5503add412187f8e75c884e9eb3420
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.00045419088564813137]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.002872243756428361]], dtype='float32').reshape([1, 1]),
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


class TestPrimitiveOp_83d3ad2fc9a30ae7d809fd58771afdcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8357bb4e351fddef99148117a3fc5989
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.0473918542265892], [-0.03837237507104874], [0.028331568464636803], [0.04863915964961052], [-0.044334761798381805], [0.07564397901296616]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_a3e30a1814a09e624a6ab20fb0730ad2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8357bb4e351fddef99148117a3fc5989
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.008855897933244705], [0.060033708810806274], [-0.0013084132224321365], [0.019272584468126297], [-0.03125490993261337], [0.010095544159412384]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.0385359562933445], [0.021661335602402687], [0.027023155242204666], [0.06791174411773682], [-0.07558967173099518], [0.08573952317237854]], dtype='float32').reshape([6, 1]),
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


class TestPrimitiveOp_4af7cfbe83f3ce4766ef52d7ec0cf6f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67d965451e79a3207d538c5bcb8f9776
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.2448769062757492]]], dtype='float32').reshape([1, 1, 1]),
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


class TestPrimitiveOp_a1ffb56d9b2148ee46c6d8c6e9c52509(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(59.74375534057617, dtype='float32').reshape([]),
            paddle.to_tensor([7.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_83a162f1f3b240296e3257805631fef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(551.640380859375, dtype='float32').reshape([]),
            paddle.to_tensor([4.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_b7ee4c9125db3583d64a449124a149ff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2076, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2076, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_31dc76deee66d9b1716cbd141f65f78d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7ee4c9125db3583d64a449124a149ff
    def get_inputs(self):
        return [
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_31dc76deee66d9b1716cbd141f65f78d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7ee4c9125db3583d64a449124a149ff
    def get_inputs(self):
        return [
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e36d5666881a073c59c06ff43527afd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(-136058.125, dtype='float32').reshape([]),
            paddle.to_tensor([0.21965822577476501], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cbc695fd90dc1ab6536a10c9361a84e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(118218.140625, dtype='float32').reshape([]),
            paddle.to_tensor([0.21965822577476501], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_342a40a82793426809b281626d583c13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_477836b2efc66ca1e0c3afd9e02f7ad3
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.19212619960308075], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_a2b77f7355684d731efdbf5faaa9b9da(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4642, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[4642, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0a06b1112571ab985d2094ecc73d3085(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2b77f7355684d731efdbf5faaa9b9da
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0a06b1112571ab985d2094ecc73d3085(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2b77f7355684d731efdbf5faaa9b9da
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e9d2415a42032aab12853ed353f4ddb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(-420767.84375, dtype='float32').reshape([]),
            paddle.to_tensor([0.03204566612839699], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f64ae646aed64c70feac18047b6415b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(264153.40625, dtype='float32').reshape([]),
            paddle.to_tensor([0.03204566612839699], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_593458e6ca2d218b4bc0ad8058a68b9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7dc936e5c979ff7ec0d7be99b1868efb
    def get_inputs(self):
        return [
            paddle.to_tensor([315.3331604003906], dtype='float32').reshape([1]),
            paddle.to_tensor(2434.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_971ac9ecddfbe4df829676e4e39a0c1a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1047, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1047, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b6a7532677334e72bd51836b301cf47a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_971ac9ecddfbe4df829676e4e39a0c1a
    def get_inputs(self):
        return [
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b6a7532677334e72bd51836b301cf47a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_971ac9ecddfbe4df829676e4e39a0c1a
    def get_inputs(self):
        return [
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_675932f35c6cdf1e28a5dfd42e0a2e3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(4041.5791015625, dtype='float32').reshape([]),
            paddle.to_tensor([0.008500613272190094], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_584cadf9aad608cbcebf41a566faf4b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(14952.5107421875, dtype='float32').reshape([]),
            paddle.to_tensor([0.008500613272190094], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_faa602b6dfe028e6001cafe0287b2264(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_148d6ac01daccbf3ef5985d4a721ccdc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.05713902786374092, 0.44475075602531433, 0.15539593994617462, 0.4389669597148895], [0.4606161415576935, 0.39743274450302124, 0.23873545229434967, 0.2039780467748642]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([[0.05209348723292351, 0.43587151169776917, 0.26826128363609314, 0.3796990215778351], [0.04524902254343033, 0.24825307726860046, 0.2502993047237396, 0.420036256313324]], dtype='float32').reshape([2, 4]),
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


class TestPrimitiveOp_a7cb177344c52d5859e25a3a919cc16e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_148d6ac01daccbf3ef5985d4a721ccdc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0803893581032753, 0.2174549549818039, 0.23238444328308105, 0.09368441253900528], [0.046390533447265625, 0.21873897314071655, 0.442493736743927, 0.45326459407806396]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([[0.22183428704738617, 0.13913291692733765, 0.05925368145108223, 0.3839888572692871], [0.32831236720085144, 0.1557861715555191, 0.2888352870941162, 0.43760159611701965]], dtype='float32').reshape([2, 4]),
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


class TestPrimitiveOp_c04bfd4ad6ae84f47c75f133ac0909ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c84f2d13b6bd2bd2f44408312c90b78
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0019853594712913036]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.0028997943736612797], [-0.01001821830868721], [0.04169686883687973], [0.004098579753190279], [0.035365574061870575]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_fb44a450d6fafaffad804924d8d7b3f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c84f2d13b6bd2bd2f44408312c90b78
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.10616093128919601], [0.030246177688241005], [-0.042821478098630905], [-0.002185745630413294], [0.09906824678182602]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.10326113551855087], [0.020227959379553795], [-0.0011246075155213475], [0.001912834239192307], [0.1344338208436966]], dtype='float32').reshape([5, 1]),
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


class TestPrimitiveOp_322126ecb4f4ab60e87b20391f8a0186(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39d9ef8c2243d1b1e1d36f72628e322b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3248838484287262], dtype='float32').reshape([1]),
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


class PrimitiveOp_153a19bacf5ce931d9ec26f10b7e7254(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2359, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2359, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d3d0c35e0427b88d4d6a4f763bff5d17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_153a19bacf5ce931d9ec26f10b7e7254
    def get_inputs(self):
        return [
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d3d0c35e0427b88d4d6a4f763bff5d17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_153a19bacf5ce931d9ec26f10b7e7254
    def get_inputs(self):
        return [
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d9884e1ce4ac6e732d21019fa25fb0d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(306549.0, dtype='float32').reshape([]),
            paddle.to_tensor([0.16684766113758087], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d70a473d97ce667030ad86078cb56f6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(133712.765625, dtype='float32').reshape([]),
            paddle.to_tensor([0.16684766113758087], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_a4ba16b3fbbfd10f6b52a401f927a042(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3049, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[3049, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_db977ba27d52d1b3b2b49a665e0f89ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4ba16b3fbbfd10f6b52a401f927a042
    def get_inputs(self):
        return [
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db977ba27d52d1b3b2b49a665e0f89ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4ba16b3fbbfd10f6b52a401f927a042
    def get_inputs(self):
        return [
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb5b191642d7d7bef80df3dd5d97c060(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(297053.78125, dtype='float32').reshape([]),
            paddle.to_tensor([0.25874385237693787], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_aba88db40ae732957093d1aae4ad7d18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(173685.953125, dtype='float32').reshape([]),
            paddle.to_tensor([0.25874385237693787], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_015b8124abbf8337a3430478629f3bd7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3806, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[3806, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6c3f882520ae94aec2b9c984e3632b02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_015b8124abbf8337a3430478629f3bd7
    def get_inputs(self):
        return [
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c3f882520ae94aec2b9c984e3632b02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_015b8124abbf8337a3430478629f3bd7
    def get_inputs(self):
        return [
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_90af3bc769a5645641188c8327bc1229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(224172.71875, dtype='float32').reshape([]),
            paddle.to_tensor([0.28673818707466125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cffbb5f025cf1e2fa580aba61239788b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(216126.78125, dtype='float32').reshape([]),
            paddle.to_tensor([0.28673818707466125], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_40aefd199fbf14bc63e3ffeb45a49de6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b264268af78d24055b659edf41a403e5
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.49490052461624146], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_239722ec4fd5e0a4956e6f88f328cb54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0928d1a760bb9d25cd90186c07cd625e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c418f27aa073d7525f8067b15d32b75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(16.85296058654785, dtype='float32').reshape([]),
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


class TestPrimitiveOp_984748341e9ce94571e7c93f1ee5113e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64508d0b7fb630d6b59223975ab7586e
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.06610178202390671], [-0.027873069047927856], [-0.000992257846519351], [-0.08162915706634521]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_ca62ba8bd2d23695a769cd910c029714(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64508d0b7fb630d6b59223975ab7586e
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.002777092158794403], [0.08546094596385956], [0.03924328088760376], [0.02453184872865677]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.06887887418270111], [0.0575878769159317], [0.03825102373957634], [-0.057097308337688446]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_1904bc6316f372d85810c31856e0147a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(3.422762155532837, dtype='float32').reshape([]),
            paddle.to_tensor([7.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_55635a1d0d6a9e7c28a3f49601afcf7a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2054, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2054, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_257ddae0ba106fb3b115c8affe3eac57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55635a1d0d6a9e7c28a3f49601afcf7a
    def get_inputs(self):
        return [
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_257ddae0ba106fb3b115c8affe3eac57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55635a1d0d6a9e7c28a3f49601afcf7a
    def get_inputs(self):
        return [
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32bdd3e232c7d98f9204d5c35c0c5b19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(110290.4375, dtype='float32').reshape([]),
            paddle.to_tensor([0.1141805425286293], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_25435360613500514db61e7961905041(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(29192.98828125, dtype='float32').reshape([]),
            paddle.to_tensor([0.1141805425286293], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_8915f2114cf7b56fa211f77acaff9e65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb2c9b3aa1051c2cf862c2846a4f657b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2569820284843445], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6a8dcdba2a8971f26126c85cafb5fa1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(35.7655029296875, dtype='float32').reshape([]),
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


class TestPrimitiveOp_e559b64be1ebed4e0e82b64fa0df871d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(235.41769409179688, dtype='float32').reshape([]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1dff81792578e149fea7d90f3aaf0dd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(130.93032836914062, dtype='float32').reshape([]),
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


class PrimitiveOp_c2a79b7b6f0844590e397e8ecfadc7ed(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4218, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[4218, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_96c8f42b275b0d01dfdd147735d8feef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2a79b7b6f0844590e397e8ecfadc7ed
    def get_inputs(self):
        return [
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_96c8f42b275b0d01dfdd147735d8feef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2a79b7b6f0844590e397e8ecfadc7ed
    def get_inputs(self):
        return [
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6a426f96103ecab778523b537694c5b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(452501.5, dtype='float32').reshape([]),
            paddle.to_tensor([0.431630402803421], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4c6cb9f45f862abd3a93623290384442(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(239527.5625, dtype='float32').reshape([]),
            paddle.to_tensor([0.431630402803421], dtype='float32').reshape([1]),
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