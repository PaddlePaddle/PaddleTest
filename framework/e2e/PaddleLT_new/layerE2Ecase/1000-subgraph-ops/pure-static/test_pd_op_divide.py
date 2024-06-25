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


class TestPrimitiveOp_7a123945a02b758d127ae40c4f590a85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bb22f30640d4643b4dd4415d6347ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.24758173525333405]]], dtype='float32').reshape([1, 1, 1]),
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


class TestPrimitiveOp_5d7b23f07f2bcca27f84368e31ac4af0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7dc936e5c979ff7ec0d7be99b1868efb
    def get_inputs(self):
        return [
            paddle.to_tensor([1093.504638671875], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_d7f889de065f67d81f4b8277af4b708c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61a805987e1e8cb6a172636260758ae4
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[3.364809515460365e-07], [0.004743359051644802], [0.007327738683670759], [0.0010841215262189507], [0.0005325789097696543], [0.0008086962043307722]]], dtype='float32').reshape([1, 6, 1]),
        ]


class TestPrimitiveOp_028dfd60a4253c390618e028dffee7e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61a805987e1e8cb6a172636260758ae4
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.0010132318129763007], [0.0013360828161239624], [0.0011471203761175275], [0.00701554212719202], [0.00046614004531875253], [0.00040505401557311416]]], dtype='float32').reshape([1, 6, 1]),
        ]


class TestPrimitiveOp_0b8dc384fae9f3063fa6c331101b9b0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61a805987e1e8cb6a172636260758ae4
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.036982256919145584], [0.1339544802904129], [0.16232439875602722], [0.0698326900601387], [0.012032266706228256], [0.015515975654125214]]], dtype='float32').reshape([1, 6, 1]),
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


class TestPrimitiveOp_18147fce7b6f4f203f1ed5c6714ac0ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(9.460820198059082, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ea0399610ff653e1c00395aff998dbfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(2.448735475540161, dtype='float32').reshape([]),
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


class PrimitiveOp_088ee7c4aeea88051cba480ea460739b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1787, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1787, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1517b0008ee8b8ed34fa2f5e1dec1e37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_088ee7c4aeea88051cba480ea460739b
    def get_inputs(self):
        return [
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1517b0008ee8b8ed34fa2f5e1dec1e37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_088ee7c4aeea88051cba480ea460739b
    def get_inputs(self):
        return [
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_02c1b0e2d0b9b10db9dc07eabbd51960(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(-1475840.25, dtype='float32').reshape([]),
            paddle.to_tensor([0.31148865818977356], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7483e1202c8ca719bc175df181c9477d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(101453.9296875, dtype='float32').reshape([]),
            paddle.to_tensor([0.31148865818977356], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a716bc5aec022adeddb63629c0e11326(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(941.78173828125, dtype='float32').reshape([]),
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


class TestPrimitiveOp_f93afac4b2e873c3a20b3748d00d4136(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6d673aa53487facfa88844d66a76af7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.06997948884963989], [-0.005269330460578203], [0.0798363983631134], [-0.021746868267655373], [0.08402508497238159], [-0.17223131656646729], [-0.08678125590085983], [0.09186724573373795], [0.04042743518948555]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_762defadca5057f2b7c0e0658b24f334(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6d673aa53487facfa88844d66a76af7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.02165216952562332], [0.060427434742450714], [-0.07387766987085342], [0.02036166749894619], [-0.04735314100980759], [0.16495314240455627], [0.0727800726890564], [-0.07924720644950867], [0.05938579514622688]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.09163165837526321], [0.0551581047475338], [0.005958727095276117], [-0.0013852004194632173], [0.036671943962574005], [-0.007278168108314276], [-0.014001179486513138], [0.012620036490261555], [0.09981323033571243]], dtype='float32').reshape([9, 1]),
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


class TestPrimitiveOp_9f2d9a4a9ecd5e0ece94ced6ab0b4548(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51de3ec31ab1ea3848e4c265af096298
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.02507392317056656], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_41e724f8526c09d01f396fdf670fc280(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5585, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[5585, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b94b50b80bb418175e8cf4610e10c6ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41e724f8526c09d01f396fdf670fc280
    def get_inputs(self):
        return [
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b94b50b80bb418175e8cf4610e10c6ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41e724f8526c09d01f396fdf670fc280
    def get_inputs(self):
        return [
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9be40e289b7a3c38dce62725c56469dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(2349.897705078125, dtype='float32').reshape([]),
            paddle.to_tensor([0.2957422733306885], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d636d7ebced0a35675c02630241bb47c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(4001.39794921875, dtype='float32').reshape([]),
            paddle.to_tensor([0.2957422733306885], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_33d271c2c909433b4199764153be454b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e39c526fd111fcd8d7384d75c9d26791
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 0.0, 0.0, -0.0, 0.0027979901060462, 0.0], dtype='float32').reshape([6]),
            paddle.to_tensor([0.000540392822586, 0.0014772490831092, 0.0039475420489907265, 0.08117253333330154, 0.057784318923950195, -0.004151618108153343], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_1c0fcdb3d67b28c5031ad232fa696146(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e39c526fd111fcd8d7384d75c9d26791
    def get_inputs(self):
        return [
            paddle.to_tensor([0.011163422837853432, 0.015458540059626102, 0.011761974543333054, 0.017370354384183884, 0.02683277428150177, 0.08598946034908295], dtype='float32').reshape([6]),
            paddle.to_tensor([0.05080551654100418, 0.12006769329309464, 0.19976107776165009, 0.1921696960926056, 0.18699021637439728, 0.28608667850494385], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_72a87bc251e8c1e5f3825f8cb8724670(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e39c526fd111fcd8d7384d75c9d26791
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.001577332615852356, -0.4028489887714386, 0.012965023517608643, 0.07334579527378082, 0.09598633646965027, -0.028074711561203003], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.3425990641117096, -0.003667004406452179, 0.3044762909412384, -0.07204306125640869, 0.21650105714797974, 0.147877499461174], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_0a3a75a93a0e95d7632997187115fa1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e39c526fd111fcd8d7384d75c9d26791
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.26374387741088867, 0.30216220021247864, -0.007320582866668701, 0.3714943528175354, 0.1509227305650711, 0.34571707248687744], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.2880426347255707, -0.1752910614013672, -0.37320035696029663, 0.23272651433944702, 0.263718843460083, -0.11338463425636292], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_eb5abeac3bf4b00f3054b2392b5ce939(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e39c526fd111fcd8d7384d75c9d26791
    def get_inputs(self):
        return [
            paddle.to_tensor([0.22001045942306519, 2.7541074752807617, 0.00021332524192985147, 1.3211688995361328, 0.004256388638168573, 0.4607751965522766], dtype='float32').reshape([6]),
            paddle.to_tensor([1.22001051902771, 3.7541074752807617, 1.000213384628296, 2.321168899536133, 0.9558351039886475, 1.4607751369476318], dtype='float32').reshape([6]),
        ]


class PrimitiveOp_af283c997628a0e876286db6d9c88574(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1774, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1774, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_750ac62a65581a5b02e69fb2b6efc419(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af283c997628a0e876286db6d9c88574
    def get_inputs(self):
        return [
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_750ac62a65581a5b02e69fb2b6efc419(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af283c997628a0e876286db6d9c88574
    def get_inputs(self):
        return [
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d69fbf6fdcaed3f507ebf2fa81f289f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(122856.25, dtype='float32').reshape([]),
            paddle.to_tensor([0.21838654577732086], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f85c21d9fc95aa0a5fe9163cf5e634ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(104888.4609375, dtype='float32').reshape([]),
            paddle.to_tensor([0.21838654577732086], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fd5beab60af5502d7fc9a687c659d263(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(950.1805419921875, dtype='float32').reshape([]),
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


class PrimitiveOp_ba9e0ede6c8c2296202235c01d55b834(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1501, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1501, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f167e3469715f8e8f1ae0dac913e6ebc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba9e0ede6c8c2296202235c01d55b834
    def get_inputs(self):
        return [
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f167e3469715f8e8f1ae0dac913e6ebc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba9e0ede6c8c2296202235c01d55b834
    def get_inputs(self):
        return [
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3917b2123054b0a42f7687ba550eebad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(231496.34375, dtype='float32').reshape([]),
            paddle.to_tensor([0.49129176139831543], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5c4bffc8642329f3e337cbb2a0fe0eab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(85227.21875, dtype='float32').reshape([]),
            paddle.to_tensor([0.49129176139831543], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_c8ebcc883762b53d463a19af48dee253(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f45f300d98a23fb80ce3b79163737f4c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.2393227219581604], [0.24333104491233826]]], dtype='float32').reshape([1, 2, 1]),
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


class TestPrimitiveOp_55ba4175789d8029552a2e43aff6eefe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b5503add412187f8e75c884e9eb3420
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.09001787006855011]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_50ebd66de36d8836c717a4df13ffc0cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b5503add412187f8e75c884e9eb3420
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.13609255850315094]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.04607468843460083]], dtype='float32').reshape([1, 1]),
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


class TestPrimitiveOp_7c80e4496b803e90bf5d46063819e22b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8357bb4e351fddef99148117a3fc5989
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.015077769756317139], [-0.12219679355621338], [-0.012837361544370651], [0.030545789748430252], [-0.13230396807193756], [0.09591057151556015]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_8922e7adb0755cfe742a53a7d4cef0d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8357bb4e351fddef99148117a3fc5989
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.028937416151165962], [0.1493632048368454], [0.025826100260019302], [0.12856201827526093], [0.12986059486865997], [-0.11888052523136139]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.013859646394848824], [0.02716641128063202], [0.012988737784326077], [0.15910780429840088], [-0.0024433706421405077], [-0.022969955578446388]], dtype='float32').reshape([6, 1]),
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


class TestPrimitiveOp_b9cd74480c029ab3c1d8569399a5ae3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67d965451e79a3207d538c5bcb8f9776
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.24200409650802612]]], dtype='float32').reshape([1, 1, 1]),
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


class TestPrimitiveOp_364a90253d5b5da4f69e4eddcc9b6552(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(58.96194839477539, dtype='float32').reshape([]),
            paddle.to_tensor([7.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6481ce1ee90ba6ed55bd47c36b4eb7b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(555.4700927734375, dtype='float32').reshape([]),
            paddle.to_tensor([4.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_26cddb6e0cd64da33e04f63cb2244a96(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2049, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2049, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_36d8c73497142ed19310c4ca92ade845(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26cddb6e0cd64da33e04f63cb2244a96
    def get_inputs(self):
        return [
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36d8c73497142ed19310c4ca92ade845(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26cddb6e0cd64da33e04f63cb2244a96
    def get_inputs(self):
        return [
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0737ca904cbdd790d870fd6ba145a674(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(-3177827.25, dtype='float32').reshape([]),
            paddle.to_tensor([0.04190996289253235], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_75f9e9d533063e572d1bfb6e2db135bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(116533.5546875, dtype='float32').reshape([]),
            paddle.to_tensor([0.04190996289253235], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_4a20d28f360b6fe5ba931e42d0b658c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_477836b2efc66ca1e0c3afd9e02f7ad3
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10028515011072159], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_dce3bc5e56a0f7fcd95aabdcd7089947(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4634, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[4634, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_84c825ac100b0ac52a46844c16d59cf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dce3bc5e56a0f7fcd95aabdcd7089947
    def get_inputs(self):
        return [
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_84c825ac100b0ac52a46844c16d59cf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dce3bc5e56a0f7fcd95aabdcd7089947
    def get_inputs(self):
        return [
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6b6c722ecabf62aa8b804a80016296f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(1315657.0, dtype='float32').reshape([]),
            paddle.to_tensor([0.4335251748561859], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_720a2697b8495727fdd0200fa047df50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(263521.25, dtype='float32').reshape([]),
            paddle.to_tensor([0.4335251748561859], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_308638644a2ba9365e37904db508f0a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7dc936e5c979ff7ec0d7be99b1868efb
    def get_inputs(self):
        return [
            paddle.to_tensor([301.5934753417969], dtype='float32').reshape([1]),
            paddle.to_tensor(2434.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_05100f2d069ad501598ce3a8750df681(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1000, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1000, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ba87c014aa2e7bf484c6f6bb8f36a3f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05100f2d069ad501598ce3a8750df681
    def get_inputs(self):
        return [
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba87c014aa2e7bf484c6f6bb8f36a3f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05100f2d069ad501598ce3a8750df681
    def get_inputs(self):
        return [
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_48fb36cd71055f434cb2b1fec229d331(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(12044.37890625, dtype='float32').reshape([]),
            paddle.to_tensor([0.3569713830947876], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d7f46988513d86cd4f67ccad9a1fc31e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(14193.6259765625, dtype='float32').reshape([]),
            paddle.to_tensor([0.3569713830947876], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_491c78a4c9427698fe47fd12659a07e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_148d6ac01daccbf3ef5985d4a721ccdc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.03989022970199585, 0.3549172878265381, 0.1134934052824974, 0.12638020515441895], [0.08357265591621399, 0.35652869939804077, 0.4814459979534149, 0.32377564907073975]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([[0.06264213472604752, 0.09819921851158142, 0.43385860323905945, 0.011553235352039337], [0.2233925759792328, 0.09674104303121567, 0.1804552972316742, 0.09168608486652374]], dtype='float32').reshape([2, 4]),
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


class TestPrimitiveOp_04a42c61b73ddd048c3147c9af8d5c31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_148d6ac01daccbf3ef5985d4a721ccdc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.19880980253219604, 0.3413020372390747, 0.4176804721355438, 0.4627673625946045], [0.4189187288284302, 0.13442544639110565, 0.19865697622299194, 0.13020436465740204]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([[0.4141802191734314, 0.18722666800022125, 0.018161043524742126, 0.017565440386533737], [0.18142472207546234, 0.3162723779678345, 0.12586869299411774, 0.0905841737985611]], dtype='float32').reshape([2, 4]),
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


class TestPrimitiveOp_8434e9e3c6fe245f3399506bb6b07ba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c84f2d13b6bd2bd2f44408312c90b78
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.019095581024885178], [0.04465426132082939], [-0.010076267644762993], [-0.05274929478764534], [0.05720999091863632]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_25cb7ec515ddc40a99ae4f583356c7ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c84f2d13b6bd2bd2f44408312c90b78
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0337679460644722], [-0.014103660359978676], [0.016030343249440193], [0.04114902764558792], [-0.023052599281072617]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.01467236690223217], [0.030550600960850716], [0.005954075139015913], [-0.011600268073379993], [0.034157391637563705]], dtype='float32').reshape([5, 1]),
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


class TestPrimitiveOp_0962e081a4f654c2d55e5b7efdbe49dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39d9ef8c2243d1b1e1d36f72628e322b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.22922326624393463], dtype='float32').reshape([1]),
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


class PrimitiveOp_17c47882cbd24e7cb7eb5ceb433eb581(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2382, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2382, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7114f8449b22d0bab3763f79ef8b80a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17c47882cbd24e7cb7eb5ceb433eb581
    def get_inputs(self):
        return [
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7114f8449b22d0bab3763f79ef8b80a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17c47882cbd24e7cb7eb5ceb433eb581
    def get_inputs(self):
        return [
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3fed6c80db0db89ba9ed46bdbeb82528(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(-1316173.625, dtype='float32').reshape([]),
            paddle.to_tensor([0.17531058192253113], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8cdfb1a82ab1d1ead9438e641d18dd47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(135447.40625, dtype='float32').reshape([]),
            paddle.to_tensor([0.17531058192253113], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_1baa61b21b037a7bbae778f67598eb2a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2976, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2976, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_54761441e935738b24371587d4e3d4fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1baa61b21b037a7bbae778f67598eb2a
    def get_inputs(self):
        return [
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_54761441e935738b24371587d4e3d4fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1baa61b21b037a7bbae778f67598eb2a
    def get_inputs(self):
        return [
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ce19571d93b5a50dc798a964b101883(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(1133.439453125, dtype='float32').reshape([]),
            paddle.to_tensor([0.18871943652629852], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6a3b7bd2c54ceeba471aefd963df09cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(169166.015625, dtype='float32').reshape([]),
            paddle.to_tensor([0.18871943652629852], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_9b9a2efe889d4d1b16c3b85291fbeb63(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3753, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[3753, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6bf5d0a9df65cea46c1aa15563aee225(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b9a2efe889d4d1b16c3b85291fbeb63
    def get_inputs(self):
        return [
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6bf5d0a9df65cea46c1aa15563aee225(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b9a2efe889d4d1b16c3b85291fbeb63
    def get_inputs(self):
        return [
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1276f0811d88baa1332e633cfa782149(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(92650.859375, dtype='float32').reshape([]),
            paddle.to_tensor([0.15429574251174927], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3a594aac38c5981a4eaa15d69c16873e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(213343.84375, dtype='float32').reshape([]),
            paddle.to_tensor([0.15429574251174927], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_945f7d9d2d5a94ccf6f442abb615b630(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b264268af78d24055b659edf41a403e5
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.30823713541030884], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_239722ec4fd5e0a4956e6f88f328cb54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0928d1a760bb9d25cd90186c07cd625e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8d0f0e441b2273b15d544d1f60b0123a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(15.295470237731934, dtype='float32').reshape([]),
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


class TestPrimitiveOp_9e1cdab38da00f4cab9be87150e15136(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64508d0b7fb630d6b59223975ab7586e
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0006502432515844703]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.0048088133335113525], [-0.038444582372903824], [0.005450582131743431], [0.0406600721180439]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_4cb343bcb22b1dfe2ae0b4e0e40f32e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64508d0b7fb630d6b59223975ab7586e
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.037636082619428635], [-0.026234228163957596], [-0.04159950464963913], [0.009962759912014008]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.04244489595293999], [-0.06467881053686142], [-0.03614892438054085], [0.05062283203005791]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_eaa5762ee154b48d47bffe1eff236fd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(4.514017105102539, dtype='float32').reshape([]),
            paddle.to_tensor([7.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_dd4b2c47586b0aea9941e1235e3588b4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1995, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1995, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_17d0030fc915ca348ee0bf8d09d14772(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd4b2c47586b0aea9941e1235e3588b4
    def get_inputs(self):
        return [
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_17d0030fc915ca348ee0bf8d09d14772(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd4b2c47586b0aea9941e1235e3588b4
    def get_inputs(self):
        return [
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3501aa53d4a1baab60ed3615a5daad48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(9186.4501953125, dtype='float32').reshape([]),
            paddle.to_tensor([0.023556096479296684], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_532d1c7fcc30dd5a3a85e3015dcf6129(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(28449.55078125, dtype='float32').reshape([]),
            paddle.to_tensor([0.023556096479296684], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_37642c4f78c6f28b1ddbcef9fa370de3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb2c9b3aa1051c2cf862c2846a4f657b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.30248215794563293], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f9de3986c5711bb66e03b09a2217273a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(35.4149055480957, dtype='float32').reshape([]),
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


class TestPrimitiveOp_e024d7a195c4b89e3c1dbe2634daf600(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(227.82875061035156, dtype='float32').reshape([]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_162e7b9b99e1651180840b86ab302376(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(135.2602996826172, dtype='float32').reshape([]),
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


class TestPrimitiveOp_e0dfe3a84d222919b9c04059bbaeb528(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(-40040.32421875, dtype='float32').reshape([]),
            paddle.to_tensor([0.3505551815032959], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_12a980e36c539742c9f3a22dc47ed44e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(237849.140625, dtype='float32').reshape([]),
            paddle.to_tensor([0.3505551815032959], dtype='float32').reshape([1]),
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