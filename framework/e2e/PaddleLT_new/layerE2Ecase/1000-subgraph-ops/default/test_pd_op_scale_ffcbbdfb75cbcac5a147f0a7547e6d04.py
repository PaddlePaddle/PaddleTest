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



class PrimitiveOp_5fb1fddda22e3f81cf03c8c25101635f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e17986ff5e1a02ced8fbc01da4a0cd04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fb1fddda22e3f81cf03c8c25101635f
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_0b3352d3a5b96111d97b5e8fb0972540(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, None, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6cc781baf8c7d609b5f628c4356dd689(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b3352d3a5b96111d97b5e8fb0972540
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_9e2c74c9e7e915c9c0e8ab2eeca69787(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 198, 198], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0de510d31af841de52cb7fd7926f2f57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e2c74c9e7e915c9c0e8ab2eeca69787
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 198, 198], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_56959cf90e818a62a6fb01ddddf0aa87(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.85, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6d1f086d9b8d2c82eae476820009f09c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56959cf90e818a62a6fb01ddddf0aa87
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.7233009934425354]]], [[[0.7237612009048462]]], [[[0.8888144493103027]]], [[[0.8999013304710388]]], [[[0.7951574921607971]]], [[[0.442484587430954]]], [[[0.6336157321929932]]], [[[0.5392743349075317]]], [[[0.905705451965332]]], [[[0.6507787108421326]]], [[[0.4348835051059723]]]], dtype='float32').reshape([11, 1, 1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_03ab246f0ef50d2f39d167b82cfcc661(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b64c3bd3bbc733d2c56a2b3f5cd26661(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03ab246f0ef50d2f39d167b82cfcc661
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.1764700412750244], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_1ec97bab89916775ffd60b3a1015e9d4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.875, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_887375add4673d70a056d558b3de3c60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ec97bab89916775ffd60b3a1015e9d4
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_48e27f2889f14b1cddf2732c5a76beb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03ab246f0ef50d2f39d167b82cfcc661
    def get_inputs(self):
        return [
            paddle.uniform([43, 112, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.1428600549697876], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_4b9908f0f4dbc186c67d7a4948572047(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_93f88b0aa4088651a41110046cb3d887(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9908f0f4dbc186c67d7a4948572047
    def get_inputs(self):
        return [
            paddle.to_tensor([[[1.1459380388259888]], [[1.0836372375488281]], [[1.4133071899414062]], [[1.0376310348510742]], [[1.4087706804275513]], [[1.2365107536315918]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9848c12e501341b53f70b96fe4121f6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9908f0f4dbc186c67d7a4948572047
    def get_inputs(self):
        return [
            paddle.to_tensor([[[1.1748725175857544]], [[1.6387921571731567]], [[1.3030327558517456]], [[1.1259150505065918]], [[1.1273354291915894]], [[1.1191586256027222]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_a7d7822d1fed62091719529c0ea5eb7c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e2f9a318b4df84ec2c93219af64e7d82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7d7822d1fed62091719529c0ea5eb7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_9af9938a4076cf462823909dc39afd33(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_23c3f07a62f43c3736ba117a5ad9b76a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9af9938a4076cf462823909dc39afd33
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_9600213b2b5203ea1935105a0554dbf7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e30f76dbb2a05d5c1b945c25d54ac46f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9600213b2b5203ea1935105a0554dbf7
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_4526b35f2d10614b12d93f6b96e2ec89(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_97f9d1937249a96b510c38034ed3bb7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4526b35f2d10614b12d93f6b96e2ec89
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_52d4dd59305a9f8794d7b34f7542cc33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_97f9d1937249a96b510c38034ed3bb7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4526b35f2d10614b12d93f6b96e2ec89
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_52d4dd59305a9f8794d7b34f7542cc33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b2342d6bed12006e2a9f4bb467835265(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4526b35f2d10614b12d93f6b96e2ec89
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ade2bcd145584cdcb79b6c9762a29444(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b2342d6bed12006e2a9f4bb467835265(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4526b35f2d10614b12d93f6b96e2ec89
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ade2bcd145584cdcb79b6c9762a29444(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8c436de5f23112c51a72421541a4cae6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4526b35f2d10614b12d93f6b96e2ec89
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_083b4d3efaf3cf209b1d805420a3eb94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8c436de5f23112c51a72421541a4cae6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4526b35f2d10614b12d93f6b96e2ec89
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_083b4d3efaf3cf209b1d805420a3eb94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_ac811c2a72ca0524f50c83bd6556b347(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d4be05710f09a3247c85113e727cf995(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac811c2a72ca0524f50c83bd6556b347
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.40612277388572693]], [[0.25845444202423096]], [[0.19602322578430176]], [[0.062916100025177]], [[0.35893920063972473]], [[0.04905514419078827]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2a8d30c30cbf705344a178e44c32ea4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac811c2a72ca0524f50c83bd6556b347
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.3443303406238556]], [[0.08050503581762314]], [[0.07664460688829422]], [[0.2265135794878006]], [[0.49640756845474243]], [[0.08292944729328156]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1325927e29b0a19a8f4df9b34fedff23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac811c2a72ca0524f50c83bd6556b347
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.3396678864955902]], [[0.40272408723831177]], [[0.3331949710845947]], [[0.1334526687860489]], [[0.06490188837051392]], [[0.13626453280448914]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.20000000298023224], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8d039430858b895cdc8262d332fbc70c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac811c2a72ca0524f50c83bd6556b347
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.494846373796463]], [[0.11729379743337631]], [[0.4344201982021332]], [[0.17414167523384094]], [[0.35329610109329224]], [[0.1349538415670395]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.20000000298023224], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_8b4e3ffb4da18304ddf9f1e56baa2416(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d3b937caf8329c221b280f7349e76763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b4e3ffb4da18304ddf9f1e56baa2416
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bc8d759f87da47ff32888a9f034110a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b4e3ffb4da18304ddf9f1e56baa2416
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_e897d27951d2dca0e85ec5790a6e446b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -0.0555556, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f293295354cab5a1d106e54572e37e26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e897d27951d2dca0e85ec5790a6e446b
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_867adfec7bf1f3263c962f4e72e6a16b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7d7822d1fed62091719529c0ea5eb7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bfd9704bc62bff43da8d30fd736d853b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9af9938a4076cf462823909dc39afd33
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_d6160d3322b1c6bb28cb3cdb54d99274(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1fe708bd73d4b101b6638dc024895f02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6160d3322b1c6bb28cb3cdb54d99274
    def get_inputs(self):
        return [
            paddle.to_tensor([[1]], dtype='int64').reshape([1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_f35a775a0ee969edad718b5c6e6cdc72(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4c9503d968ff7321eadb969351f0398d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f35a775a0ee969edad718b5c6e6cdc72
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.24566441774368286]]], dtype='float32').reshape([1, 1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b1b395808f4b687d7268b31e89ac5981(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b4e3ffb4da18304ddf9f1e56baa2416
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_600eec5c6eb2092710151ce96926bbf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b4e3ffb4da18304ddf9f1e56baa2416
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f4eb28f8c298e3ad51be783c2ca9aa34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e897d27951d2dca0e85ec5790a6e446b
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_077abd1642f8ae2cd035b1f158165f4d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f0b9532fc5f60083ddafbea8380797de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_077abd1642f8ae2cd035b1f158165f4d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6eeec5e95591aec5cade1dd0cb822a84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_077abd1642f8ae2cd035b1f158165f4d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_a56cf8864334847d664a770104d8eb8d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_315fc85e711e6cb1055e47b136811654(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a56cf8864334847d664a770104d8eb8d
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1340a7a747f69aee87a7b8bf061bd355(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a56cf8864334847d664a770104d8eb8d
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_0cedebdb00063d3794413a6eafc373c9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d92c791abc3d5c712e117ff573a0fb67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cedebdb00063d3794413a6eafc373c9
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0], dtype='float32').reshape([24]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8566c68ac1ab29b03cebc624d3c4cb0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cedebdb00063d3794413a6eafc373c9
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0], dtype='float32').reshape([24]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c57063572a467078a3dba1f53f09c037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9908f0f4dbc186c67d7a4948572047
    def get_inputs(self):
        return [
            paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_0f50d8022ff78548f62eb858288990c7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_83ddda25aadd6b491bcb000a27f9e8cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f50d8022ff78548f62eb858288990c7
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6eeec5e95591aec5cade1dd0cb822a84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_077abd1642f8ae2cd035b1f158165f4d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_cfbbb74e96a9a3cd73df6e3db19f6071(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c28fe059bbf5f090e657d416897f3d0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cfbbb74e96a9a3cd73df6e3db19f6071
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1340a7a747f69aee87a7b8bf061bd355(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a56cf8864334847d664a770104d8eb8d
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_49320ad40a0d29048d7502d846f4a69d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4b613ac899315e5f21be8112d25f831e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49320ad40a0d29048d7502d846f4a69d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0], dtype='float32').reshape([24]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e8bb63c00c2a8a6c7c25d22c576bddfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cedebdb00063d3794413a6eafc373c9
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5], dtype='float32').reshape([24]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3d05b79bed41695c7b5d47461684d4e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7d7822d1fed62091719529c0ea5eb7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_0884743459f4915d82904c5015982b8f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ed7daf5c52586b646382609a0a3bd810(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0884743459f4915d82904c5015982b8f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b8261c8c6ca769da010cfa2db17721ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9af9938a4076cf462823909dc39afd33
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2e8f76faee089e0f282e89a36d57920c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7d7822d1fed62091719529c0ea5eb7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_26fdd96db2d35f2e0c886d370f91665b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9af9938a4076cf462823909dc39afd33
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_699166084feb39f20e7855e1829a3ad4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1f6b597685475c25cf3d6f026b2126fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(1106.72607421875, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ea05122c5e9879d0778858cbd4e18676(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(136.95657348632812, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a0a0d0e0a715f51e160d328dbdd194b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(5.513241767883301, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_90a9d192480b64079293c7f00caa0fa6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4d109ce0ea91c450b7fb1d081df32aaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90a9d192480b64079293c7f00caa0fa6
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.0014888138975948095], [0.00013513748126570135], [0.055184073746204376], [0.0012555427383631468], [0.013625239953398705], [7.0934411269263364e-06]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_01247a4f103c15f0c492b53b101c982b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90a9d192480b64079293c7f00caa0fa6
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.0022997611667960882], [4.731779335997999e-05], [0.02808908186852932], [0.0054276022128760815], [0.004598488099873066], [7.206148438854143e-05]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_406921816d9b8047f7a20b58c83cb1fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9908f0f4dbc186c67d7a4948572047
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-6.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0387b756b5d783217779fce719f8a210(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac811c2a72ca0524f50c83bd6556b347
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.13436663150787354], [0.060790590941905975], [0.4693793058395386], [0.10528217256069183], [0.06731704622507095], [0.0008048239978961647]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([0.08333329856395721], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fc3495232619c0fa5546bb2dec976637(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac811c2a72ca0524f50c83bd6556b347
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.01119721494615078], [0.005065880250185728], [0.039114926010370255], [0.008773510344326496], [0.005609751679003239], [6.706863496219739e-05]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([6.28318977355957], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0eb178077bec3b218591dd11e26e7e5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f35a775a0ee969edad718b5c6e6cdc72
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.07035422325134277], [0.03182988613843918], [0.2457665055990219], [0.055125631392002106], [0.035247135907411575], [0.00042140495497733355]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_401eb4b4f029154dd5cae15374049d6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7d7822d1fed62091719529c0ea5eb7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f54dbe4b543b68af451ff90aae3c9977(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0884743459f4915d82904c5015982b8f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d6c4e66c1cf8b817caccf768d3a029d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9af9938a4076cf462823909dc39afd33
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7d1ddc0752730cecf252203843ce3330(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.9125, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5cbabc747d63e51543ee4d1cf2ef613d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d1ddc0752730cecf252203843ce3330
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a8d629e4a64c31e5d1639e6975555c78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03ab246f0ef50d2f39d167b82cfcc661
    def get_inputs(self):
        return [
            paddle.uniform([43, 80, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0958900451660156], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04ce4e62b5989f2e4feb6e797661a79f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b3352d3a5b96111d97b5e8fb0972540
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_6b2435231cb0b72f5bf9fbe721fbabae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1838b37d2467fe344ae29914aff0b72c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b2435231cb0b72f5bf9fbe721fbabae
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([16]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_445064f4218542290a789c0e504134f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.to_tensor([1.8891524076461792, 2.0357344150543213, 2.1746408939361572, 2.0617501735687256, 2.168100357055664, 1.9462820291519165, 2.033737897872925, 2.1954946517944336, 1.896557331085205, 1.9024568796157837, 2.05024790763855, 2.172727346420288, 2.066915988922119, 2.084320306777954, 2.0626168251037598, 2.1053898334503174], dtype='float32').reshape([16]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_264b8c0e771a6a529d70fdf524d13c5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(1.9941877126693726, dtype='float32').reshape([]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e7518ea9bb600c3b0d162fb95909efaf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7d7822d1fed62091719529c0ea5eb7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e61ef3da4ad9bc37ec4fa86115bafb6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9af9938a4076cf462823909dc39afd33
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ce769e2d9ddc9dcca08295563e25e911(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9600213b2b5203ea1935105a0554dbf7
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_d2f5c701ebc4b7993ff4593c39bce75f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ff5f136c54017cba95692f4cb11afdfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2f5c701ebc4b7993ff4593c39bce75f
    def get_inputs(self):
        return [
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ff5f136c54017cba95692f4cb11afdfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2f5c701ebc4b7993ff4593c39bce75f
    def get_inputs(self):
        return [
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ed719d6509efa0049e18c1b5567f91e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9600213b2b5203ea1935105a0554dbf7
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e2f9a318b4df84ec2c93219af64e7d82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7d7822d1fed62091719529c0ea5eb7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b490fb656ba4c746f2f1385973a74007(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0884743459f4915d82904c5015982b8f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_93757ca75bbd8aec72362eb50ab4460d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9af9938a4076cf462823909dc39afd33
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_aaf9eb01860bac2d07fefe1b5ccbb9f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7d7822d1fed62091719529c0ea5eb7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e190541fd3702c50394cadd50f3ade20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9af9938a4076cf462823909dc39afd33
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d11bac01a12aaf7bebe0e809f8c46d6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(37.6960334777832, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_6b6147eb9e3d4540fa230991620e9845(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 12, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dad529b6ef3865104188f816494f1348(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b6147eb9e3d4540fa230991620e9845
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 577, 577], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1e29c838c5790e072d13daaae567420d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.uniform([150], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_d09c1660cd2ad2a76fdc80dbc1d5524f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2442936b9c953fd1f69ff4e93406132d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d09c1660cd2ad2a76fdc80dbc1d5524f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
            paddle.to_tensor([80.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d561cacbea2024cbf00637fc4cac2b45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d09c1660cd2ad2a76fdc80dbc1d5524f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_be78ed9a480ce1763a6d3fd601c278c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_16abb8e7dedafb807d1765c213c91c98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7d7822d1fed62091719529c0ea5eb7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9304f6466b4dbbfc74753ae7db50618a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0884743459f4915d82904c5015982b8f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c9345b8b02d6dd7df77c45cd51df2a1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9af9938a4076cf462823909dc39afd33
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_06f90143b8ab9d69120ff1073a9dc131(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 16384, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5e27661c69d9ccf9bb533a4ef4ff8a4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06f90143b8ab9d69120ff1073a9dc131
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_95864ebc6da496802f63c6f82174b6ba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e5b6b989495d3cd93221a83a8298ee10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95864ebc6da496802f63c6f82174b6ba
    def get_inputs(self):
        return [
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e5b6b989495d3cd93221a83a8298ee10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95864ebc6da496802f63c6f82174b6ba
    def get_inputs(self):
        return [
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_bce202c470bbbf07a610c93dd97477d6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d550e7abd6e07a08ad635e53214b18e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bce202c470bbbf07a610c93dd97477d6
    def get_inputs(self):
        return [
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_52c1c16ac8d019b900f1c735a7d7334d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ecb7075ba4f7c8edce94fd92f084f850(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_52c1c16ac8d019b900f1c735a7d7334d
    def get_inputs(self):
        return [
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_86299bdcdf08bf954fc6518b813b5b7f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3f0312a430838ad646894cb8b1b685a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86299bdcdf08bf954fc6518b813b5b7f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1841, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_626de8f444c822bb894fb31988f051d3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3494a271016f76cd2c57944ee5cc44c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_626de8f444c822bb894fb31988f051d3
    def get_inputs(self):
        return [
            paddle.uniform([1841, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_71bd1f02664b7f7df6bc8e3f5c3bb387(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d796288782576e72f38492afbba2608d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71bd1f02664b7f7df6bc8e3f5c3bb387
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1841, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d796288782576e72f38492afbba2608d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71bd1f02664b7f7df6bc8e3f5c3bb387
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1841, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_740344bef65e797e3e21a0302b109d98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_294d4137d4a600513d2d3bf1244ea829(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9600213b2b5203ea1935105a0554dbf7
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b62f27a751c354fb184e8bf044b6f6f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ec97bab89916775ffd60b3a1015e9d4
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.2507723569869995]]], [[[0.10466217994689941]]], [[[0.15142057836055756]]], [[[0.03664936497807503]]], [[[0.8497947454452515]]], [[[0.6683487892150879]]], [[[0.3721596300601959]]], [[[0.6742473244667053]]], [[[0.1959022730588913]]], [[[0.8804839253425598]]], [[[0.12412010133266449]]]], dtype='float32').reshape([11, 1, 1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fbb726a707dca0611ee336838469bb56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03ab246f0ef50d2f39d167b82cfcc661
    def get_inputs(self):
        return [
            paddle.uniform([11, 112, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.1428600549697876], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_8b9c3e62b05bfc6cd607c3af196c561e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.95, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a14abf40de568ab5fb0bade3b5737460(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b9c3e62b05bfc6cd607c3af196c561e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.3441122770309448]]], [[[0.8362777233123779]]], [[[0.7145991921424866]]], [[[0.9102576971054077]]], [[[0.2564704418182373]]], [[[0.06279069185256958]]], [[[0.36559057235717773]]], [[[0.44241034984588623]]], [[[0.17329442501068115]]], [[[0.8176327347755432]]], [[[0.8911646604537964]]]], dtype='float32').reshape([11, 1, 1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6bef5181b1df4219849534bfbf0eee38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03ab246f0ef50d2f39d167b82cfcc661
    def get_inputs(self):
        return [
            paddle.uniform([11, 40, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0526299476623535], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b84d1d4d14461693f90bf3cb41c429e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d09c1660cd2ad2a76fdc80dbc1d5524f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
            paddle.to_tensor([9.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0c52286567942f96debd9a3725775c52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d09c1660cd2ad2a76fdc80dbc1d5524f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_624639cc8a3eb832403ad92361d618f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(158.56060791015625, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6e380d8486ddb30b08eadad8e2a5a9e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(3.4983372688293457, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b7f77b803843c70f3ea3a911fb00834f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7d7822d1fed62091719529c0ea5eb7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_18c5ff7fa6912d3d5cf38b57199f5d4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9af9938a4076cf462823909dc39afd33
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8c06a1af5d94843b480dd89fa203450c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7d7822d1fed62091719529c0ea5eb7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fe00de0e146d53d5801d77118ad3234f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0884743459f4915d82904c5015982b8f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_de0604c4db8b7ee04ef6ac7ed236f55b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9af9938a4076cf462823909dc39afd33
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_9d60d914d23aa26f64d03d6aa0d6ebe8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d145e8dfcdb17c1c20c3ba3dfc775868(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d60d914d23aa26f64d03d6aa0d6ebe8
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f9d62c01b45d450a30c199efde0d6aca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d60d914d23aa26f64d03d6aa0d6ebe8
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_a5e682b4ee939ca23901b4ad22819a94(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_02ed9f3ba80e98d820b1cec76f4a0a87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5e682b4ee939ca23901b4ad22819a94
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9930f4f202acd144cda1c40287701a9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5e682b4ee939ca23901b4ad22819a94
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_501c33c7d623297e4ad2d951c7dacf94(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ddf1ff94f97900bb515293428cf803c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_501c33c7d623297e4ad2d951c7dacf94
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype='float32').reshape([16]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fe6e68c45a2ff092290d10ebac8a663a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_501c33c7d623297e4ad2d951c7dacf94
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype='float32').reshape([16]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_deb73becced98f6c5759831c9f04d43c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9908f0f4dbc186c67d7a4948572047
    def get_inputs(self):
        return [
            paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_27d0a016da023d787152f0123accfcd8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4ae8a647da6ae930dcea153346509215(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27d0a016da023d787152f0123accfcd8
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f9d62c01b45d450a30c199efde0d6aca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d60d914d23aa26f64d03d6aa0d6ebe8
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_bcfab7f6120c6e92a0ab9645baffb419(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a3029350eb0e396db1f5593e46d7350b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcfab7f6120c6e92a0ab9645baffb419
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9930f4f202acd144cda1c40287701a9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5e682b4ee939ca23901b4ad22819a94
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_861b9f5093066d708bff81a9c78a39ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0e2e42f2673c1cf6a58724552bdbd660(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_861b9f5093066d708bff81a9c78a39ae
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype='float32').reshape([16]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cbeca4ac23fff43507123b3c66a25ddf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_501c33c7d623297e4ad2d951c7dacf94
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5], dtype='float32').reshape([16]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e9266b2f5899c43f6355f6510dc9ba9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d09c1660cd2ad2a76fdc80dbc1d5524f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
            paddle.to_tensor([80.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_abacdb10842bc3af6f4d5bd6ba572135(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d09c1660cd2ad2a76fdc80dbc1d5524f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_67cb32aa75b14e70831ae0b2bf4ef96a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(171.1174774169922, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9c439492f2b14c1c14bebd9bb9ed2830(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(2.772587776184082, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_bad0cdc814f871416a345678b7bfe23b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.975, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6503101bdb381df31e52cfb729f5bc25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bad0cdc814f871416a345678b7bfe23b
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_62242bc83acb197f32a9ac052c607e12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03ab246f0ef50d2f39d167b82cfcc661
    def get_inputs(self):
        return [
            paddle.uniform([43, 24, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0256400108337402], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e30f76dbb2a05d5c1b945c25d54ac46f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9600213b2b5203ea1935105a0554dbf7
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c96d146e7e14cd35b1f3c88a1a06d037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7d7822d1fed62091719529c0ea5eb7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_afe32a543886c60ee5313ae762a6504f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9af9938a4076cf462823909dc39afd33
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_783b94dcaff2575db6fc483d4b0b0045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7d7822d1fed62091719529c0ea5eb7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_af41558ba46e07d2fdfbb8e64e20699d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0884743459f4915d82904c5015982b8f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2efb0252f404ed2af5c0e13260ee0fdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9af9938a4076cf462823909dc39afd33
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0025143f5b51814a8843d390d8dc3763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03ab246f0ef50d2f39d167b82cfcc661
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.01010000705719], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_f1811779b2a5c416b22e37c688b06c8c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2858c9dbbcefe64e5095c3974024fffa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1811779b2a5c416b22e37c688b06c8c
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.04838085174560547], [-0.056969888508319855], [0.065152607858181], [-0.0019278854597359896], [0.0031600119546055794], [-0.13211898505687714], [0.06693486869335175], [-0.03826824575662613], [-0.05094216763973236]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_738879501b04a20cde93907f9d67b3bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1811779b2a5c416b22e37c688b06c8c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.04462939873337746], [-0.09026241302490234], [-0.0763222873210907], [0.0036242941860109568], [0.025886045768857002], [-0.053869184106588364], [-0.042909879237413406], [-0.009765285067260265], [-0.008294115774333477]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_d1255f7a4bcebf297eee2ebc8e49a136(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7ee56845ca9b65c3dedc3aaebecbfd96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1255f7a4bcebf297eee2ebc8e49a136
    def get_inputs(self):
        return [
            paddle.to_tensor([[-2.0840578079223633], [-0.3688414990901947], [-1.8536512851715088], [-1.5319340229034424], [-0.8779259920120239], [1.4525892734527588], [-2.55989408493042], [2.91880464553833], [5.141964912414551]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7f0161b7466ae1bf0665beb78e06f63a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b4e3ffb4da18304ddf9f1e56baa2416
    def get_inputs(self):
        return [
            paddle.to_tensor([[3.0840578079223633], [1.368841528892517], [2.853651285171509], [2.5319340229034424], [1.877925992012024], [-0.4525892734527588], [3.55989408493042], [-1.91880464553833], [-4.141964912414551]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_185c4bd51cbdb666f8dad51190c25b6b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, None, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b1cf5fc4d4db97834e906f00af2e687d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_185c4bd51cbdb666f8dad51190c25b6b
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 640, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_592d3e85cc1caac6a44383fafad4ca7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(11643.5546875, dtype='float32').reshape([]),
            paddle.to_tensor([0.09090910106897354], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f197bffe0881384b85acd02d32cb9717(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(1058.505126953125, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5c2dbca1f65535e765bab3ad63f09ca7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09989378601312637], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59d499f99df2c80c68786c22cbc60018(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9908f0f4dbc186c67d7a4948572047
    def get_inputs(self):
        return [
            paddle.to_tensor([[[1.2462061643600464]], [[1.2301290035247803]], [[1.2096160650253296]], [[1.4589134454727173]], [[1.000787615776062]], [[1.4105521440505981]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9ce1bea9ff2498410d0278f5279fa6a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9908f0f4dbc186c67d7a4948572047
    def get_inputs(self):
        return [
            paddle.to_tensor([[[1.2946494817733765]], [[1.3159765005111694]], [[1.5503513813018799]], [[1.039616346359253]], [[1.035115361213684]], [[1.4364465475082397]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_a5979570d19df229baf6bb203b1dfcfa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c6f26cc1e54d0586c662c29dcbc6e6c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5979570d19df229baf6bb203b1dfcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_5d5f9bd71fabbc72e3807e6e4c080102(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d796e582b20df45278478b4d31a1d023(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d5f9bd71fabbc72e3807e6e4c080102
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c6f26cc1e54d0586c662c29dcbc6e6c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5979570d19df229baf6bb203b1dfcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6f5834c8c53ec44715795e36c02054b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e2c74c9e7e915c9c0e8ab2eeca69787
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 198, 198], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1ca6b42458b4806c256078ba804b28e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95864ebc6da496802f63c6f82174b6ba
    def get_inputs(self):
        return [
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1ca6b42458b4806c256078ba804b28e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95864ebc6da496802f63c6f82174b6ba
    def get_inputs(self):
        return [
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_541ad26b781a8a24de82aa716f50808c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bce202c470bbbf07a610c93dd97477d6
    def get_inputs(self):
        return [
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_33f3748f1bd71accee98f09a4a3452d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_52c1c16ac8d019b900f1c735a7d7334d
    def get_inputs(self):
        return [
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f7359632a8f3a2040c371375cbd745d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86299bdcdf08bf954fc6518b813b5b7f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[5562, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7fd677d453310dbc01e4961287b51407(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_626de8f444c822bb894fb31988f051d3
    def get_inputs(self):
        return [
            paddle.uniform([5562, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0ce98d14ab911e642363ea8104aeae57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71bd1f02664b7f7df6bc8e3f5c3bb387
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[5562, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0ce98d14ab911e642363ea8104aeae57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71bd1f02664b7f7df6bc8e3f5c3bb387
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[5562, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7b0be87a58810bed51f2f3b6995e9ef2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a761e369aef83a6e069e91991e66dcdf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b0be87a58810bed51f2f3b6995e9ef2
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_db5e0ebd14c954fb41dd58c9152aab40(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1000], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6d9109d3a8a42bea255990c030d9268a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db5e0ebd14c954fb41dd58c9152aab40
    def get_inputs(self):
        return [
            paddle.uniform([86, 1000], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2bf7fc4525c2009bca9f02b1a44768b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db5e0ebd14c954fb41dd58c9152aab40
    def get_inputs(self):
        return [
            paddle.uniform([54, 1000], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_82c9db5f05b6b5957493ca53fa23194d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.8375, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e76b706b1c7f53c4d0fa1bd148dd3885(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_82c9db5f05b6b5957493ca53fa23194d
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d2d05e1023b1e57160df9e8cb7661b35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03ab246f0ef50d2f39d167b82cfcc661
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.1940300464630127], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ace32b194e874303d02445db773b938(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(149.71188354492188, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5302736b346cb314ed4bb70ffe0a71bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(5.598021984100342, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_09137a2422dbcc3eb5123a531ec92b37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b2435231cb0b72f5bf9fbe721fbabae
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_91e82a42b0128509f7aa63d940d07b0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8c6854a8c7d2b532a398ee55d2c9a77b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(6.323805332183838, dtype='float32').reshape([]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_de113d8b2e1c1ae4aab1077d48da10d1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c93813d29f9a8e93f9bb10874b9b6c05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de113d8b2e1c1ae4aab1077d48da10d1
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_16e2c3f6ec434a69754b5507465f4090(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_66b17a750f1f16c4d6de34969d106ebb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4943939745426178], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0c9b3411274d35cdb4d62c0887b5e469(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.06618824601173401], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_280550037450f5b1099a09a7720b304c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.to_tensor([1.000994324684143], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ce96d9a0d68fc3060fc58e54e14e3552(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b4e3ffb4da18304ddf9f1e56baa2416
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_770870f0c3eb202c2798f34d4778477d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b4e3ffb4da18304ddf9f1e56baa2416
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_73903025a84bb1199ca5c2baf2812e9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e897d27951d2dca0e85ec5790a6e446b
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5e593e4e0ba05ef8ec50e42323203454(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(173.71539306640625, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ffe67abc870fa2f36a4c59adc90096ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(46.99455261230469, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_5a145d1a0738c95ec3acc2366fecd6d2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cfe9c9da192980655b3bbc3958969276(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a145d1a0738c95ec3acc2366fecd6d2
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0025143f5b51814a8843d390d8dc3763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03ab246f0ef50d2f39d167b82cfcc661
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.01010000705719], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d11bfb9f20b3ab802579dbbae887b684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.to_tensor([0.21679174900054932, 0.5053405165672302, 0.4347398281097412, 0.7754498720169067, 0.3926175534725189, 0.6659517288208008], dtype='float32').reshape([6]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_000a6b014e87130766895a0f1c62621a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5648379325866699, 0.20716355741024017, 0.3146982192993164, 0.5089221000671387, 0.5010338425636292, 0.4067259430885315], dtype='float32').reshape([6]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e23f623e1157c5e464e11ec94df4b3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5244249105453491, 0.12409630417823792, 0.10802936553955078, 0.5870891809463501, 0.1490015983581543, 0.7131940722465515], dtype='float32').reshape([6]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_46396b827b5278e1114200e4b3daa320(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4350762367248535, 0.574497401714325, 0.294285386800766, 0.3647545278072357, 0.33535027503967285, 0.4974139332771301], dtype='float32').reshape([6]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_a0c770b38c84ad14480ac00f4f1e66dc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fb46b90b6ce93c2b7823a524e1ec339d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0c770b38c84ad14480ac00f4f1e66dc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.01093385647982359, -0.020309938117861748, -0.0014530873158946633, 0.011614530347287655, -0.027645012363791466, -0.017154740169644356], dtype='float32').reshape([6]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f38b1fe818c64aa490ea202b1f0c5c37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0c770b38c84ad14480ac00f4f1e66dc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.02786906436085701, 0.07007032632827759, 0.0267891027033329, 0.014066009782254696, 0.021699944511055946, 0.002614037599414587], dtype='float32').reshape([6]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8341b69573c53a4443c6ca547d3df105(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0c770b38c84ad14480ac00f4f1e66dc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.008555368520319462, 0.1272517740726471, 0.0688125342130661, 0.022314582020044327, 0.22995047271251678, 0.07298903167247772], dtype='float32').reshape([6]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cc3a12dd3993f64b7bf27e58a86d8e87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.to_tensor([1.459782600402832, 1.0664626359939575, 0.5840408802032471, -0.401843786239624, -0.36295560002326965, -0.1388222873210907], dtype='float32').reshape([6]),
            paddle.to_tensor([0.4052850008010864], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_737b031602b630b137ac0eec78de8030(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bca00df1bc3a2a3bac2fab5179894e42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_737b031602b630b137ac0eec78de8030
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, -0.0, -0.0, 0.0, -0.0, 0.0], dtype='float32').reshape([6]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_12bcb961e65b2a85f7090fb6b84a83b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0c770b38c84ad14480ac00f4f1e66dc
    def get_inputs(self):
        return [
            paddle.to_tensor([1.8636482954025269, 1.460947871208191, 1.1382442712783813, 1.065444827079773, 1.0533909797668457, 1.0078104734420776], dtype='float32').reshape([6]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_21c67f10a8a76388e74636727cce22d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.to_tensor([4.657724380493164, 1.6960781812667847, 1.4060958623886108, 1.6343705654144287, 1.0970739126205444, 1.0358747243881226], dtype='float32').reshape([6]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3faad4aab3be355dd78476cfdd150895(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(1.9212028980255127, dtype='float32').reshape([]),
            paddle.to_tensor([10.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6cc781baf8c7d609b5f628c4356dd689(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b3352d3a5b96111d97b5e8fb0972540
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8565ee2f877ea7c2f2f45561123a2219(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95864ebc6da496802f63c6f82174b6ba
    def get_inputs(self):
        return [
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8565ee2f877ea7c2f2f45561123a2219(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95864ebc6da496802f63c6f82174b6ba
    def get_inputs(self):
        return [
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_773ec8ee8a3885b3623636470e2ce645(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bce202c470bbbf07a610c93dd97477d6
    def get_inputs(self):
        return [
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4095d8fa1807ad0784d01833bdc4785e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_52c1c16ac8d019b900f1c735a7d7334d
    def get_inputs(self):
        return [
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_65c056ebc63d5e16701c299e8d903143(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86299bdcdf08bf954fc6518b813b5b7f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1734, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_00e57cf0aa8058598cf3f1eeb8553729(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_626de8f444c822bb894fb31988f051d3
    def get_inputs(self):
        return [
            paddle.uniform([1734, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_8587ae20adda5efb17d114694560bae8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 2, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_66514c9f4de22099dbc4b91fb2ee8f28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8587ae20adda5efb17d114694560bae8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1734, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_66514c9f4de22099dbc4b91fb2ee8f28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8587ae20adda5efb17d114694560bae8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1734, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_4d4dc2965698bbfff26e52d07448b60b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_64192781fbcaa98161663d6866606c40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d4dc2965698bbfff26e52d07448b60b
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_09d7e73a12a83f037b40a3cc61ed53f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d4dc2965698bbfff26e52d07448b60b
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 256, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_20c3f0b2ff1c6ab81d1f6e0a3e6c83a8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 100, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e9cd492b29af7c6418613ddd9f387ba1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c3f0b2ff1c6ab81d1f6e0a3e6c83a8
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 100, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_601870bc7897ec307c0858093e81d4d0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 100, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_539137be84ed82dc399e18cf9b8d5d60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_601870bc7897ec307c0858093e81d4d0
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 100, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_25841ae379d99c94c42cdb4a1ff405ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac811c2a72ca0524f50c83bd6556b347
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.06928415596485138]], [[0.14566513895988464]], [[0.4753703773021698]], [[0.026734229177236557]], [[0.4941661059856415]], [[0.47734639048576355]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_49ef2af6ffa13c21608be6bff0f86fdf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac811c2a72ca0524f50c83bd6556b347
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.11425824463367462]], [[0.2625088393688202]], [[0.4443609118461609]], [[0.2234448343515396]], [[0.14226004481315613]], [[0.41878029704093933]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1aa3d28598cf143ae5abe1f2aba879d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac811c2a72ca0524f50c83bd6556b347
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.10753093659877777]], [[0.10612644255161285]], [[0.393868088722229]], [[0.4168972074985504]], [[0.01984567753970623]], [[0.4527571499347687]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.20000000298023224], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_21293056894ec49565ea3c38030564b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac811c2a72ca0524f50c83bd6556b347
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.3928699493408203]], [[0.4585893154144287]], [[0.06145929545164108]], [[0.2693466246128082]], [[0.0032234126701951027]], [[0.0999179258942604]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.20000000298023224], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_740344bef65e797e3e21a0302b109d98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d54e923ecb8d59fd818e26058cf344fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9600213b2b5203ea1935105a0554dbf7
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c0a883bce688aaeda8e4204b164a6f5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2644745111465454], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a823a802629793a68d7077cb5e165320(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1626177579164505], dtype='float32').reshape([1]),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8a53b66acf9dcb1c7012b87d9834711f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.to_tensor([0.19221025705337524], dtype='float32').reshape([1]),
            paddle.to_tensor([0.05000000074505806], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ceed4979b7bffd85870a207cc3ec8e61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03ab246f0ef50d2f39d167b82cfcc661
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_4da469a78e23afb577317277efe9d4a5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_91baf4d9417c6c3f4ebf28d60e947408(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4da469a78e23afb577317277efe9d4a5
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6fd85e2f6c491cde854054fa504c38c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4da469a78e23afb577317277efe9d4a5
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_55bdce66a71497878845c53c67a53385(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b9a6b620aa02183b1840db39e6748fdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55bdce66a71497878845c53c67a53385
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_25c1d5b276d03f22431fa8dcff64dadf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55bdce66a71497878845c53c67a53385
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_dd1adeaa64b38c4276ec6e8e6855f9e1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e8301af09c3dfdf12a7c6433c29cf8ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd1adeaa64b38c4276ec6e8e6855f9e1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0], dtype='float32').reshape([20]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e6272f45d7d8e3c9c5e52202061795fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd1adeaa64b38c4276ec6e8e6855f9e1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0], dtype='float32').reshape([20]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6143ef1df0d8b6277213569bdf601ae2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9908f0f4dbc186c67d7a4948572047
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_06be4122bb08a6e0bd2606d613e178d2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4600f14674436750dcd617c3c163a429(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06be4122bb08a6e0bd2606d613e178d2
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6fd85e2f6c491cde854054fa504c38c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4da469a78e23afb577317277efe9d4a5
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_65d2aa88cdd67118a0fdf9114743747c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d6a71a1eebafb839ec764cbd14de1ad6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65d2aa88cdd67118a0fdf9114743747c
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_25c1d5b276d03f22431fa8dcff64dadf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55bdce66a71497878845c53c67a53385
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7be50029f9d563d7329e5f69ea219b57(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_20711143a1c28f279ec09c24f44f689e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7be50029f9d563d7329e5f69ea219b57
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0], dtype='float32').reshape([20]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4682705c4e12f2c7829926030fb34325(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd1adeaa64b38c4276ec6e8e6855f9e1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5], dtype='float32').reshape([20]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9c289c6bf98831e366a0f5abf1877d5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b2435231cb0b72f5bf9fbe721fbabae
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([24]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_10c1b8dc6300cb8e4049ee1665b68931(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.to_tensor([1.8821864128112793, 2.167982339859009, 2.0223448276519775, 2.1801035404205322, 2.1758694648742676, 2.077899932861328, 2.1036360263824463, 2.2210097312927246, 2.0764780044555664, 2.246574878692627, 2.12399959564209, 2.048368215560913, 2.1236016750335693, 2.0511302947998047, 1.9186406135559082, 2.1834330558776855, 2.0810956954956055, 2.135617733001709, 2.0576720237731934, 2.2162697315216064, 2.0051450729370117, 2.1424484252929688, 2.0627388954162598, 2.1449594497680664], dtype='float32').reshape([24]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d74cfd44a9d04d73e4dd178e86da93d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(3.119081497192383, dtype='float32').reshape([]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a6da0d87199375e308186bd47dfb055b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.05022953450679779], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_31bbae978276fea4acb68c61ca5a7aa1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d73e4301a484dcb0a1ac067e301123b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31bbae978276fea4acb68c61ca5a7aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.18873784236799077], dtype='float64').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_97f9d1937249a96b510c38034ed3bb7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4526b35f2d10614b12d93f6b96e2ec89
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_52d4dd59305a9f8794d7b34f7542cc33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_97f9d1937249a96b510c38034ed3bb7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4526b35f2d10614b12d93f6b96e2ec89
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_52d4dd59305a9f8794d7b34f7542cc33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b2342d6bed12006e2a9f4bb467835265(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4526b35f2d10614b12d93f6b96e2ec89
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ade2bcd145584cdcb79b6c9762a29444(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b2342d6bed12006e2a9f4bb467835265(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4526b35f2d10614b12d93f6b96e2ec89
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ade2bcd145584cdcb79b6c9762a29444(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8c436de5f23112c51a72421541a4cae6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4526b35f2d10614b12d93f6b96e2ec89
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_083b4d3efaf3cf209b1d805420a3eb94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8c436de5f23112c51a72421541a4cae6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4526b35f2d10614b12d93f6b96e2ec89
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_083b4d3efaf3cf209b1d805420a3eb94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2e8f76faee089e0f282e89a36d57920c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7d7822d1fed62091719529c0ea5eb7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_20c21067792b528a1827526af1d242bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0884743459f4915d82904c5015982b8f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1110dfca8f3ada39b04df45d165eca89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9af9938a4076cf462823909dc39afd33
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8c06a1af5d94843b480dd89fa203450c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7d7822d1fed62091719529c0ea5eb7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cf8fcb20789d854f5543046dc3c9ad6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9af9938a4076cf462823909dc39afd33
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c4cf17956e7b4a9194e6081f5dff5529(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9600213b2b5203ea1935105a0554dbf7
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7dab959bf0627d782fe4b2658f57ab20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95864ebc6da496802f63c6f82174b6ba
    def get_inputs(self):
        return [
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7dab959bf0627d782fe4b2658f57ab20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95864ebc6da496802f63c6f82174b6ba
    def get_inputs(self):
        return [
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_10911aed46327bbfbb62c8f7036201b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bce202c470bbbf07a610c93dd97477d6
    def get_inputs(self):
        return [
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d7e32119ab913c34da7f3f4f1ee35216(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_52c1c16ac8d019b900f1c735a7d7334d
    def get_inputs(self):
        return [
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c4c114b488c83a8958a4d63d129fb4c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86299bdcdf08bf954fc6518b813b5b7f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1541, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_79fa136616dea050cbf5d6e7534c9fcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_626de8f444c822bb894fb31988f051d3
    def get_inputs(self):
        return [
            paddle.uniform([1541, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a82c033762e22727f30b497be8e4a6df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71bd1f02664b7f7df6bc8e3f5c3bb387
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1541, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a82c033762e22727f30b497be8e4a6df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71bd1f02664b7f7df6bc8e3f5c3bb387
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1541, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d81c32f7aa52db9280450cc5503ee38c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6160d3322b1c6bb28cb3cdb54d99274
    def get_inputs(self):
        return [
            paddle.to_tensor([[9]], dtype='int64').reshape([1, 1]),
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_75b9fc8531fc9ba6f0c82aaf1044e64e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f35a775a0ee969edad718b5c6e6cdc72
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.2480098158121109], [0.24436978995800018]]], dtype='float32').reshape([1, 2, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e7518ea9bb600c3b0d162fb95909efaf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7d7822d1fed62091719529c0ea5eb7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ab87a2cb4c2a7ac823f6ed9e3093b308(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0884743459f4915d82904c5015982b8f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cb21f42b89a8cedb8b0cd8fc6992ae5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9af9938a4076cf462823909dc39afd33
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_78320b05a98b71c6a03ce110c09e33f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b2435231cb0b72f5bf9fbe721fbabae
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0], dtype='int64').reshape([4]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a1fbfdede604aef6cc0eddadcbfb6936(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.to_tensor([2.10895037651062, 2.0647668838500977, 2.2277159690856934, 2.106340169906616], dtype='float32').reshape([4]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_aae1d034ffcc36b7f48d2c4145c14f3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(0.6357869505882263, dtype='float32').reshape([]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_867adfec7bf1f3263c962f4e72e6a16b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7d7822d1fed62091719529c0ea5eb7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e9fc4c8e1b73c8136c01199c77f6c61e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0884743459f4915d82904c5015982b8f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_68d4381413a9785d4a9b37690160c12e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9af9938a4076cf462823909dc39afd33
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2680cdf71f3e3592496fa526b3a1a1ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(146.47018432617188, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3ccc4046426df149e066f95664d2d195(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(3.6282477378845215, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_135b76817887df4be6aaa6410eadea89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(147.9531707763672, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_748b099bd3eb06e8213e5a82d6e33efe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(2.5891292095184326, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_b63e01909a92a47391412fe2026d9342(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 19, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8b90850b78985fe9140cd0748c0cccff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b63e01909a92a47391412fe2026d9342
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 512, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_4e4922968922a1a6756c98fbf770d107(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 6, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_33f511ec78b9460e6f2e4d464a99cab1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e4922968922a1a6756c98fbf770d107
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1025, 1025], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9e8c24298953daf902972bf0a8f3a40a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(183.76397705078125, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e95ca0061ab4f30392882fac14a41626(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(7.942829132080078, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_aa6c958eab2f01205c96ce4ce55f7abf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1811779b2a5c416b22e37c688b06c8c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.008202476426959038]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bafc4d9db80c43fab9064f9f3963fe3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1811779b2a5c416b22e37c688b06c8c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.008620435371994972]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7eb1852d00fa6cf1da387f5952e79bcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1255f7a4bcebf297eee2ebc8e49a136
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.048484668135643005]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_305d5d4967f3c37d1c03d1c286780e34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b4e3ffb4da18304ddf9f1e56baa2416
    def get_inputs(self):
        return [
            paddle.to_tensor([[1.0484846830368042]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b9da2bd4560f055feb659d0d7da50106(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1811779b2a5c416b22e37c688b06c8c
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.09975834935903549], [0.039119549095630646], [0.059811756014823914], [-0.10562919825315475], [-0.003041415009647608], [0.03176024928689003]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8f5ad2c6ea2b85b5e9199e291ab62faf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1811779b2a5c416b22e37c688b06c8c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.01078457199037075], [0.01947149634361267], [0.004929106682538986], [-0.00237811217084527], [-0.0018230847781524062], [0.049544066190719604]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ede87b012867befe090ad6f097955a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1255f7a4bcebf297eee2ebc8e49a136
    def get_inputs(self):
        return [
            paddle.to_tensor([[-10.25009822845459], [1.009067416191101], [11.134400367736816], [43.4172477722168], [0.6682796478271484], [-0.3589494824409485]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1de02c0160c162ecccfa319530deccd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b4e3ffb4da18304ddf9f1e56baa2416
    def get_inputs(self):
        return [
            paddle.to_tensor([[11.25009822845459], [-0.009067416191101074], [-10.134400367736816], [-42.4172477722168], [0.33172035217285156], [1.3589494228363037]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8e9072c643b8ad1d5775a9eabbf48a50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bad0cdc814f871416a345678b7bfe23b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.036384981125593185]]], [[[0.7357426881790161]]], [[[0.6908688545227051]]], [[[0.605592668056488]]], [[[0.13173624873161316]]], [[[0.7235406637191772]]], [[[0.43877923488616943]]], [[[0.8839250206947327]]], [[[0.24989020824432373]]], [[[0.04156850650906563]]], [[[0.4723178744316101]]]], dtype='float32').reshape([11, 1, 1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_84a5bbb7c70e8a167a89b541c0fcfac9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03ab246f0ef50d2f39d167b82cfcc661
    def get_inputs(self):
        return [
            paddle.uniform([11, 24, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0256400108337402], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_eb4253799ebf0e06bd17d9c6c3cc4e24(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[14], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f9b06314642ca37163bc865de1cb00de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb4253799ebf0e06bd17d9c6c3cc4e24
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0], dtype='float32').reshape([14]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_22d219f2e2f925ceeab15052197f93a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[14], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_42cc2e3d5a213a00c04435ab1a42241c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22d219f2e2f925ceeab15052197f93a3
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5], dtype='float32').reshape([14]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_8781a49b754f4f6dc72561c7411d7b86(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -80, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_35fd2c7f013c9bae0fdf5341d35500c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8781a49b754f4f6dc72561c7411d7b86
    def get_inputs(self):
        return [
            paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_35fd2c7f013c9bae0fdf5341d35500c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8781a49b754f4f6dc72561c7411d7b86
    def get_inputs(self):
        return [
            paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_f6ad56a52320f0d5d337853311377310(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 80, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c460d486f78aa1670978b8cfeef44ed8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6ad56a52320f0d5d337853311377310
    def get_inputs(self):
        return [
            paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c460d486f78aa1670978b8cfeef44ed8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6ad56a52320f0d5d337853311377310
    def get_inputs(self):
        return [
            paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_6a431a288c5e02bca6e00dc99999b2bb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e8870d58c2da208a06f75dbe08f7a8be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a431a288c5e02bca6e00dc99999b2bb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0], dtype='float32').reshape([28]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_9df834a23d21b275d29b029f7e35ffc8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6f989e96e5da43c7c29b8d1a5f9a6671(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9df834a23d21b275d29b029f7e35ffc8
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5], dtype='float32').reshape([28]),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_4ec84098a9049f3d015d58dbc05364bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -40, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6b72878d2aba136664a8421b2a020c58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ec84098a9049f3d015d58dbc05364bf
    def get_inputs(self):
        return [
            paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6b72878d2aba136664a8421b2a020c58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ec84098a9049f3d015d58dbc05364bf
    def get_inputs(self):
        return [
            paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_35b2ffe01b66d5a568669cc7743fd846(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 40, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_24e9e48ca1dd51fc6803657c442e42b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35b2ffe01b66d5a568669cc7743fd846
    def get_inputs(self):
        return [
            paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_24e9e48ca1dd51fc6803657c442e42b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35b2ffe01b66d5a568669cc7743fd846
    def get_inputs(self):
        return [
            paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7fb046410ef3da911fff37e026697f08(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[56], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9b41f6458ec266d3a4c0c4ff462a6893(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7fb046410ef3da911fff37e026697f08
    def get_inputs(self):
        return [
            paddle.uniform([56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c2f06c1ce5ea54062fbe88bd372bc5ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[56], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_53dca9fae4ba139076d1e9fa3d113921(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2f06c1ce5ea54062fbe88bd372bc5ef
    def get_inputs(self):
        return [
            paddle.uniform([56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_2cf30a250438920dad73c05744325945(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -20, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_93d9d2a1cb6fd0349ee32a2581b7d75a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cf30a250438920dad73c05744325945
    def get_inputs(self):
        return [
            paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_93d9d2a1cb6fd0349ee32a2581b7d75a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cf30a250438920dad73c05744325945
    def get_inputs(self):
        return [
            paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_44a84e4e0d9da031122c6b658bd046e3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 20, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7367164f3b920f2ba9779434c29f8609(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44a84e4e0d9da031122c6b658bd046e3
    def get_inputs(self):
        return [
            paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7367164f3b920f2ba9779434c29f8609(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44a84e4e0d9da031122c6b658bd046e3
    def get_inputs(self):
        return [
            paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_fd3b8b240747c1ba620654de03e03565(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b65cf46c219f98dabb80d0ef1a5e3995(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd3b8b240747c1ba620654de03e03565
    def get_inputs(self):
        return [
            paddle.to_tensor(4, dtype='int32').reshape([]),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1dd55dbae13519ee09cac831657039c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd3b8b240747c1ba620654de03e03565
    def get_inputs(self):
        return [
            paddle.to_tensor(16, dtype='int32').reshape([]),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c2ed26e2ba00baa8920d63d9bec6ce85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd3b8b240747c1ba620654de03e03565
    def get_inputs(self):
        return [
            paddle.to_tensor(13, dtype='int32').reshape([]),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c2ed26e2ba00baa8920d63d9bec6ce85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd3b8b240747c1ba620654de03e03565
    def get_inputs(self):
        return [
            paddle.to_tensor(13, dtype='int32').reshape([]),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2eacc2b9e00bbdc1b2370b036ff35226(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6160d3322b1c6bb28cb3cdb54d99274
    def get_inputs(self):
        return [
            paddle.to_tensor([[3]], dtype='int64').reshape([1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0376386e5a7da745bbac900bb56960bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f35a775a0ee969edad718b5c6e6cdc72
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.24596427381038666]]], dtype='float32').reshape([1, 1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_5f8d070f71a65ad111798d37dd9aa7c9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 2048, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_629440ef52d0af9055747dadad79c854(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f8d070f71a65ad111798d37dd9aa7c9
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_239bb937b9af4dfd04295a5da65ed0c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
            paddle.to_tensor([0.0015625000232830644], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7884cd7a42c3622d8e8a43d1a0189d62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(7.0, dtype='float32').reshape([]),
            paddle.to_tensor([0.0015625000232830644], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_61b397285958a88bcc1ae6227a1f6c5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a145d1a0738c95ec3acc2366fecd6d2
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b07c23842754cb17001b4b0b56f9f025(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(133.38009643554688, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_55893f5b4205cff5ba0032945eaaaa1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(2.4077696800231934, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_961f3390a1c87b8497a1ed54c7ef031e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_50d357549346f4f9831b551961a96c15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_961f3390a1c87b8497a1ed54c7ef031e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b7f77b803843c70f3ea3a911fb00834f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7d7822d1fed62091719529c0ea5eb7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_84931cefb37a1084e7388e6ba4f471c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0884743459f4915d82904c5015982b8f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_11b19dd85a69e48df00f3bedced769d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9af9938a4076cf462823909dc39afd33
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_ccf84142974138841b80fa5a1f911d0b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 197, 197], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_83a3d3ef3b9ac5ff85530edbd006834a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccf84142974138841b80fa5a1f911d0b
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 197, 197], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3d05b79bed41695c7b5d47461684d4e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7d7822d1fed62091719529c0ea5eb7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_511b0802a0e3bb7bfdd7d872984ebc00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9af9938a4076cf462823909dc39afd33
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_128f0936711e37b3d97dc1dfea138fd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(201.4234161376953, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_512e7654eec78dbcca379a678b1e3898(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(65.10199737548828, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_aea8ae797393ab8e346cadca33845a8b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 65536, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f12cf06b527595cf3a3b7da088fe876b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aea8ae797393ab8e346cadca33845a8b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1806557559666c610f35cfce03005da5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.uniform([950], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_24d5d83f5d243d5eb9a7123ab337ffb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.uniform([8816], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c0709c05deaae57c15631227825c90b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95864ebc6da496802f63c6f82174b6ba
    def get_inputs(self):
        return [
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c0709c05deaae57c15631227825c90b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95864ebc6da496802f63c6f82174b6ba
    def get_inputs(self):
        return [
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c34828f8d8815fff72805fc6360d0434(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bce202c470bbbf07a610c93dd97477d6
    def get_inputs(self):
        return [
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_edacd3d46a497bd8e0f1885fd19cd1ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_52c1c16ac8d019b900f1c735a7d7334d
    def get_inputs(self):
        return [
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3f2f5a15d7de5a90ca2a7e0c22565797(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86299bdcdf08bf954fc6518b813b5b7f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2061, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3d1fac80d761706e5ff44d934a0973cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_626de8f444c822bb894fb31988f051d3
    def get_inputs(self):
        return [
            paddle.uniform([2061, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9d8955b3d24c2b96ca5ee24f274261eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71bd1f02664b7f7df6bc8e3f5c3bb387
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2061, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9d8955b3d24c2b96ca5ee24f274261eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71bd1f02664b7f7df6bc8e3f5c3bb387
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2061, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2071a1ca77091f7af57a22dfb29fb488(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_961f3390a1c87b8497a1ed54c7ef031e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a7d453e624b9747c82791a28ca420753(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(169.83367919921875, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_76d240f68067646c7e0628da7fd872fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(5.037328243255615, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4b613ac899315e5f21be8112d25f831e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49320ad40a0d29048d7502d846f4a69d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0], dtype='float32').reshape([24]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bdbba4669cbe29d57ac7d72fbf869ae1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cedebdb00063d3794413a6eafc373c9
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5], dtype='float32').reshape([24]),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_961e0de3fa820b64087ec6031a2fd13d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2636254131793976], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f6a1dce63f777c16401c02ad924ce295(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4488471746444702], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_461ff77444d7a2c8b195831348d00865(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.17543905973434448], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7715b1bd4f9a1f8bda18d32a97f41bf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4183725118637085], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_332288d1446950a604a6b642c83c534e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.23224054276943207], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_616461bab40b42276f6eb28eafcab494(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5628452301025391], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_756c430006dbccf5e98cef238fccf0ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.48140838742256165], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_84a253374dab88c853dfd8186fdd2598(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.to_tensor([0.7383275032043457], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f811f2617f8683ff1a9a5eaa6b33649c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.08521274477243423], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_df71de8726e83d786f2fa199ca3a6b84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5216418504714966], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_639c62bfce7886a610e19ff620c5fb61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.19825448095798492], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_881fdd685e56a7b7b7a69c4d8316b749(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.to_tensor([0.22831273078918457], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ab994cd7d61b332d2da0c1525eb5d14a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2891826033592224], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3240cff71b0786be27110b6a413f4d3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5791901350021362], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_45245892683e2c74a54cbab0c7317d5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.25390130281448364], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_50135a7a197678d950c8c1da2de85554(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3454288840293884], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5eaa96dac7aeeafd43a0647290f8b1bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16e2c3f6ec434a69754b5507465f4090
    def get_inputs(self):
        return [
            paddle.to_tensor([0.27220115065574646], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5bff67b21c691fb03d9edc0c1538efbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5738576650619507], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2cf43d6b109c674cda86f42e200e23a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.to_tensor([0.24109217524528503], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7c1c3de1258cb18ee2d4db0d06481757(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.to_tensor([0.28952065110206604], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d6f67b421e2bf80d0363c2c45e51ba53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4170323610305786], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b54b541339789ede5afeb5afcf2a5c02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.to_tensor([0.7150324583053589], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_449996b1c13a28ef1a1e6fab04e4cf73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7d7822d1fed62091719529c0ea5eb7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_780cdd7169cfb16985e95afdb0c95e74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0884743459f4915d82904c5015982b8f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a8628feddcbefafdc9e8e39935c85df7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9af9938a4076cf462823909dc39afd33
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_28f3968edf2b35117064f979cfc33e24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b4e3ffb4da18304ddf9f1e56baa2416
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_10f8ba839be018a0ed94438a7fe72c3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b4e3ffb4da18304ddf9f1e56baa2416
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d99d1d5b1c3dfb5d302220f0d76f18a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e897d27951d2dca0e85ec5790a6e446b
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f0283859eb1a9d0954f93d6582ca3921(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5979570d19df229baf6bb203b1dfcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2dc12426a4a69d0335e4ea1e059b4eb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d5f9bd71fabbc72e3807e6e4c080102
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f0283859eb1a9d0954f93d6582ca3921(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5979570d19df229baf6bb203b1dfcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2bd30cdd0471ba3bfc82ab8e33d534cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95864ebc6da496802f63c6f82174b6ba
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2bd30cdd0471ba3bfc82ab8e33d534cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95864ebc6da496802f63c6f82174b6ba
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_97cf980499e061dab24485c7967717cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bce202c470bbbf07a610c93dd97477d6
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_740d3af6a1f65e01a21b1562ae0db5a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_52c1c16ac8d019b900f1c735a7d7334d
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5b5d4c8349c1f3a4965e138ba3ccd1f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86299bdcdf08bf954fc6518b813b5b7f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4642, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8c103cffedcc9843bdc226e6693a48f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_626de8f444c822bb894fb31988f051d3
    def get_inputs(self):
        return [
            paddle.uniform([4642, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5273ff431eab6dc5285f5bc371e039e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71bd1f02664b7f7df6bc8e3f5c3bb387
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4642, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5273ff431eab6dc5285f5bc371e039e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71bd1f02664b7f7df6bc8e3f5c3bb387
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4642, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04ce4e62b5989f2e4feb6e797661a79f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b3352d3a5b96111d97b5e8fb0972540
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7b68b04c901770024177f7ad0085b9ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(105.18740844726562, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_90ffd1d007062d96a1921e0e852b0f8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(303.8099365234375, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_624e92511347372c202986716273bc14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95864ebc6da496802f63c6f82174b6ba
    def get_inputs(self):
        return [
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_624e92511347372c202986716273bc14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95864ebc6da496802f63c6f82174b6ba
    def get_inputs(self):
        return [
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_745b8625ead8cabb19b56863f6c737bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bce202c470bbbf07a610c93dd97477d6
    def get_inputs(self):
        return [
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_98aaf4a452a837737f386c88f3edbe5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_52c1c16ac8d019b900f1c735a7d7334d
    def get_inputs(self):
        return [
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_86ad0b7dda09a582bd07837af93f6d81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86299bdcdf08bf954fc6518b813b5b7f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1042, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cdcfcf86e06068e3cf6ac7455b710ad3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_626de8f444c822bb894fb31988f051d3
    def get_inputs(self):
        return [
            paddle.uniform([1042, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d4f71f38cf6c0b979ca6c5f67cc36376(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71bd1f02664b7f7df6bc8e3f5c3bb387
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1042, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d4f71f38cf6c0b979ca6c5f67cc36376(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71bd1f02664b7f7df6bc8e3f5c3bb387
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1042, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_07da1b56c4289e55404bebb46f57a9ec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 32768, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_084ebfc813bb86bd734c782e4714a5a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07da1b56c4289e55404bebb46f57a9ec
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1c0cda8bc7a6c1d91d93cc0f793a9319(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03ab246f0ef50d2f39d167b82cfcc661
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-50.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_94074940be3a11df88ed07fe19fce89a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_06dc172288c86512107a8a1a8d896072(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94074940be3a11df88ed07fe19fce89a
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ded4b612ca3dba39c0b28f70ab4c1fb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(156.44692993164062, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8ee5dc9640f91cf1d1709f894eb0485e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(4.6403350830078125, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8fa83128131317e11447c75e03480dcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_185c4bd51cbdb666f8dad51190c25b6b
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 200, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_dc7d1cd873efe64dddaa7b4bbd876cdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9600213b2b5203ea1935105a0554dbf7
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_401eb4b4f029154dd5cae15374049d6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7d7822d1fed62091719529c0ea5eb7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_57dfd4766d2225f7ea06b5ad57c15dac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9af9938a4076cf462823909dc39afd33
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d5b675b6940451f6028ef3d6a1cf8233(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b4e3ffb4da18304ddf9f1e56baa2416
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.75], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da4ff8f91e8d9166456472fa17a46b5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1255f7a4bcebf297eee2ebc8e49a136
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_42f21d71ff64b1d218657a0dddb2ef5f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-08, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_048750d9b4d84e092dbf33e6662dfdae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42f21d71ff64b1d218657a0dddb2ef5f
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9dedb23c5b4ae45ebbf6c474fefc8171(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b4e3ffb4da18304ddf9f1e56baa2416
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f162b59be938a45816db2b4bca64bca9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b4e3ffb4da18304ddf9f1e56baa2416
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_048750d9b4d84e092dbf33e6662dfdae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42f21d71ff64b1d218657a0dddb2ef5f
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9dedb23c5b4ae45ebbf6c474fefc8171(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b4e3ffb4da18304ddf9f1e56baa2416
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c96d146e7e14cd35b1f3c88a1a06d037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7d7822d1fed62091719529c0ea5eb7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d206b9653466adecf735cf140da03f6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0884743459f4915d82904c5015982b8f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_238778ea1d225117b99048c1f78343f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9af9938a4076cf462823909dc39afd33
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_e9221ed9eb8592f088df67e050d92a00(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[68], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b867780487642fd52b20f1a10429a105(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9221ed9eb8592f088df67e050d92a00
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bf2b722c168d09ae34041790eacd2a72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9221ed9eb8592f088df67e050d92a00
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7b395bf4de92806cef3841fa70edf552(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[34], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d302bc2bc3414872559023fd8c4908a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b395bf4de92806cef3841fa70edf552
    def get_inputs(self):
        return [
            paddle.uniform([34], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_781554fae082d67ffe651e8bbcaa1c94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b395bf4de92806cef3841fa70edf552
    def get_inputs(self):
        return [
            paddle.uniform([34], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_0311368f5e8300e40600799b5c7bdf1c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[17], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_80fd32fa1dfcc168f53d00fef8c5a062(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0311368f5e8300e40600799b5c7bdf1c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], dtype='float32').reshape([17]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_239e617c1b7e9e8e3320d2f19203a23a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0311368f5e8300e40600799b5c7bdf1c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], dtype='float32').reshape([17]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5b82a6cdc7894b94faebbc71f54bcf6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9908f0f4dbc186c67d7a4948572047
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_33b5c6082ddd5dcd20b4f4184fd1a306(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[68], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6fb1e0bf6a77eae29c02e959fba6b838(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_33b5c6082ddd5dcd20b4f4184fd1a306
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bf2b722c168d09ae34041790eacd2a72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9221ed9eb8592f088df67e050d92a00
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7f54a117e3dfc66256bd7d727348cafd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[34], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c53755aabdef581a6e114a7f252eb9c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f54a117e3dfc66256bd7d727348cafd
    def get_inputs(self):
        return [
            paddle.uniform([34], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_781554fae082d67ffe651e8bbcaa1c94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b395bf4de92806cef3841fa70edf552
    def get_inputs(self):
        return [
            paddle.uniform([34], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_e373054e34fcada833dbc73da8ced72d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[17], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_171f720e5133572e580b9d5599827c43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e373054e34fcada833dbc73da8ced72d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], dtype='float32').reshape([17]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_49494c93cc9e2c4da8eaf1b5bab44d70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0311368f5e8300e40600799b5c7bdf1c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5], dtype='float32').reshape([17]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a761e369aef83a6e069e91991e66dcdf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b0be87a58810bed51f2f3b6995e9ef2
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a761e369aef83a6e069e91991e66dcdf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b0be87a58810bed51f2f3b6995e9ef2
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a761e369aef83a6e069e91991e66dcdf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b0be87a58810bed51f2f3b6995e9ef2
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7682808d5e6343f426546d5b1653e757(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2048, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_48af82642baeeb9a810bb4dab30f1ef4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7682808d5e6343f426546d5b1653e757
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bb4eced3f61118f9d8fa4e688be83046(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b4e3ffb4da18304ddf9f1e56baa2416
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.75], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c09a98420873a6fd5a4418d786c696d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1255f7a4bcebf297eee2ebc8e49a136
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fa98cefd4368de863014336ff220499d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42f21d71ff64b1d218657a0dddb2ef5f
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_29e1ea53efa83500df7211c196fdd1bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b4e3ffb4da18304ddf9f1e56baa2416
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5688f9fd101075db7214cae53e06c706(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b4e3ffb4da18304ddf9f1e56baa2416
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fa98cefd4368de863014336ff220499d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42f21d71ff64b1d218657a0dddb2ef5f
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_29e1ea53efa83500df7211c196fdd1bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b4e3ffb4da18304ddf9f1e56baa2416
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cb647616e10979946a889334dc408348(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1811779b2a5c416b22e37c688b06c8c
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.022827766835689545], [0.007110994309186935], [0.06232514977455139], [-0.057350385934114456], [-0.041275642812252045]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_acd43924ce5c193c073db0f8e00c4534(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1811779b2a5c416b22e37c688b06c8c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0024515173863619566], [-0.00240074354223907], [0.005132114514708519], [0.035371504724025726], [0.015316097997128963]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4f8588d71f748aceb745440c0064b32e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1255f7a4bcebf297eee2ebc8e49a136
    def get_inputs(self):
        return [
            paddle.to_tensor([[-10.311688423156738], [-3.9619967937469482], [11.144145965576172], [-2.621372699737549], [-3.6949191093444824]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_278b805643cef1e616b29cfdb6242d33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b4e3ffb4da18304ddf9f1e56baa2416
    def get_inputs(self):
        return [
            paddle.to_tensor([[11.311688423156738], [4.961997032165527], [-10.144145965576172], [3.621372699737549], [4.694919109344482]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_066290e400a309599edf8c14cfc4a104(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 8192, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3e050ad7a8443126ec3992858074fff5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_066290e400a309599edf8c14cfc4a104
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8c436de5f23112c51a72421541a4cae6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4526b35f2d10614b12d93f6b96e2ec89
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_083b4d3efaf3cf209b1d805420a3eb94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8c436de5f23112c51a72421541a4cae6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4526b35f2d10614b12d93f6b96e2ec89
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_083b4d3efaf3cf209b1d805420a3eb94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b2342d6bed12006e2a9f4bb467835265(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4526b35f2d10614b12d93f6b96e2ec89
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ade2bcd145584cdcb79b6c9762a29444(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b2342d6bed12006e2a9f4bb467835265(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4526b35f2d10614b12d93f6b96e2ec89
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ade2bcd145584cdcb79b6c9762a29444(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_97f9d1937249a96b510c38034ed3bb7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4526b35f2d10614b12d93f6b96e2ec89
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_52d4dd59305a9f8794d7b34f7542cc33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_97f9d1937249a96b510c38034ed3bb7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4526b35f2d10614b12d93f6b96e2ec89
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_52d4dd59305a9f8794d7b34f7542cc33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5141611d700ed69216f5b5f2e72f72ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4526b35f2d10614b12d93f6b96e2ec89
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype='float32').reshape([16]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f5cbda016974ee844142fb24ff82079b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5], dtype='float32').reshape([16]),
            paddle.to_tensor([64.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5141611d700ed69216f5b5f2e72f72ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4526b35f2d10614b12d93f6b96e2ec89
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype='float32').reshape([16]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f5cbda016974ee844142fb24ff82079b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5], dtype='float32').reshape([16]),
            paddle.to_tensor([64.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bc5f56dc0dda05b9a029a799112d0fb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4526b35f2d10614b12d93f6b96e2ec89
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype='float32').reshape([8]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a99287cfb36ac4da88774785208475b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], dtype='float32').reshape([8]),
            paddle.to_tensor([128.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bc5f56dc0dda05b9a029a799112d0fb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4526b35f2d10614b12d93f6b96e2ec89
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype='float32').reshape([8]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a99287cfb36ac4da88774785208475b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], dtype='float32').reshape([8]),
            paddle.to_tensor([128.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a37d1b7357ec9ea86999ee640ed05cf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f8d070f71a65ad111798d37dd9aa7c9
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_98d4a923f3851a1d3fd1a769bf8f0abd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9600213b2b5203ea1935105a0554dbf7
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3d96451fa931d16b878746c7ee8f7e98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b0be87a58810bed51f2f3b6995e9ef2
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3d96451fa931d16b878746c7ee8f7e98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b0be87a58810bed51f2f3b6995e9ef2
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3d96451fa931d16b878746c7ee8f7e98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b0be87a58810bed51f2f3b6995e9ef2
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_88c27cfd227047ef169865567f91f8a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7682808d5e6343f426546d5b1653e757
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d843facdbe21cfbc0dd5d7f8aff0cfeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5979570d19df229baf6bb203b1dfcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bbca83be97e4b19f2255d70ac362826b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d5f9bd71fabbc72e3807e6e4c080102
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d843facdbe21cfbc0dd5d7f8aff0cfeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5979570d19df229baf6bb203b1dfcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_16abb8e7dedafb807d1765c213c91c98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7d7822d1fed62091719529c0ea5eb7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4f123dd06b00a45164e972ee678e425d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9af9938a4076cf462823909dc39afd33
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_783b94dcaff2575db6fc483d4b0b0045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7d7822d1fed62091719529c0ea5eb7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6eae9e93b9cff084a224478ae527e4fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9af9938a4076cf462823909dc39afd33
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1cf814b543620e69c96663c8ed99f745(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7d7822d1fed62091719529c0ea5eb7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f442cbb8d09af876ab5d971a7ceca2c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9af9938a4076cf462823909dc39afd33
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f6be88c0cd45f8ea867ffbec927b4f8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3902689814567566], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0172467e59ba43569288c7fa55f520f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2968399226665497], dtype='float32').reshape([1]),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0c124c0d67054fd4aa9bd2a1e65d30e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.to_tensor([0.24719944596290588], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_efaf4d1759cd24c4aee9d7112d826bcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b6147eb9e3d4540fa230991620e9845
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1025, 1025], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_dc126e694af0c17c5a01a6a752f8bdbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d09c1660cd2ad2a76fdc80dbc1d5524f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
            paddle.to_tensor([9.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2634f7d2d0f369a2470a3b7d05e1c581(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d09c1660cd2ad2a76fdc80dbc1d5524f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_95b275dcdaa8e4895fd33fef8f905ea5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95864ebc6da496802f63c6f82174b6ba
    def get_inputs(self):
        return [
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_95b275dcdaa8e4895fd33fef8f905ea5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95864ebc6da496802f63c6f82174b6ba
    def get_inputs(self):
        return [
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c02dca955cb1b8fcba4bde84c4df52fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bce202c470bbbf07a610c93dd97477d6
    def get_inputs(self):
        return [
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e1d5d597af6a5c080542974fbe36de6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_52c1c16ac8d019b900f1c735a7d7334d
    def get_inputs(self):
        return [
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_33f2b5c2dda830f260ba134371366129(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86299bdcdf08bf954fc6518b813b5b7f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2369, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b682cd37893108e845ffa7fb8bda8058(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_626de8f444c822bb894fb31988f051d3
    def get_inputs(self):
        return [
            paddle.uniform([2369, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4889d05cc755f336c60aff0480e4ca2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71bd1f02664b7f7df6bc8e3f5c3bb387
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2369, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4889d05cc755f336c60aff0480e4ca2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71bd1f02664b7f7df6bc8e3f5c3bb387
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2369, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_935d44e7873ac9fba9f7846c1403e021(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95864ebc6da496802f63c6f82174b6ba
    def get_inputs(self):
        return [
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_935d44e7873ac9fba9f7846c1403e021(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95864ebc6da496802f63c6f82174b6ba
    def get_inputs(self):
        return [
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_80dd19393bcc25ab93b863eb37b8f413(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bce202c470bbbf07a610c93dd97477d6
    def get_inputs(self):
        return [
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c142366c1fecd4d1e87bddd1426df60d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_52c1c16ac8d019b900f1c735a7d7334d
    def get_inputs(self):
        return [
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_07ac79c1441359fb2f6340d081ca677b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86299bdcdf08bf954fc6518b813b5b7f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3054, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4571e9a161892d2fca6c3a8cfbfd755d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_626de8f444c822bb894fb31988f051d3
    def get_inputs(self):
        return [
            paddle.uniform([3054, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d2b24dce06b49b48eb1e84439ad50136(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71bd1f02664b7f7df6bc8e3f5c3bb387
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3054, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d2b24dce06b49b48eb1e84439ad50136(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71bd1f02664b7f7df6bc8e3f5c3bb387
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3054, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_951faa1c4b9b693a9d35be2a373a0fc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95864ebc6da496802f63c6f82174b6ba
    def get_inputs(self):
        return [
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_951faa1c4b9b693a9d35be2a373a0fc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95864ebc6da496802f63c6f82174b6ba
    def get_inputs(self):
        return [
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a2ef733a03aa818586ec06d3273bb1f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bce202c470bbbf07a610c93dd97477d6
    def get_inputs(self):
        return [
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cd63c0381331e8aca5852309862ddcf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_52c1c16ac8d019b900f1c735a7d7334d
    def get_inputs(self):
        return [
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5480eed344b41380d300571de83667d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86299bdcdf08bf954fc6518b813b5b7f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3819, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_35e476e775076acd8a921ce7dd6d0347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_626de8f444c822bb894fb31988f051d3
    def get_inputs(self):
        return [
            paddle.uniform([3819, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a8181b730723e243e323bd60ff14a2c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71bd1f02664b7f7df6bc8e3f5c3bb387
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3819, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a8181b730723e243e323bd60ff14a2c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71bd1f02664b7f7df6bc8e3f5c3bb387
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3819, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_26dac50cfdf7f51de15995599ccc9dc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5979570d19df229baf6bb203b1dfcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([64.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_80dedf1aea8a4e982786ae05219b1f07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d5f9bd71fabbc72e3807e6e4c080102
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_26dac50cfdf7f51de15995599ccc9dc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5979570d19df229baf6bb203b1dfcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([64.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7394c593833e499967e8bb817ad53a5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b4e3ffb4da18304ddf9f1e56baa2416
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f14feaa466f7473d969bc71467f70e94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b4e3ffb4da18304ddf9f1e56baa2416
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_eb8b90f56d461f2a94724f0fb78fbe19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e897d27951d2dca0e85ec5790a6e446b
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_5d487cf1b57f98d236eed82908f0fbbf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.925, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a6f541ac359a5c8997a44c86c643d9e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d487cf1b57f98d236eed82908f0fbbf
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.46358126401901245]]], [[[0.10091772675514221]]], [[[0.6349579095840454]]], [[[0.9504490494728088]]], [[[0.6027831435203552]]], [[[0.026347285136580467]]], [[[0.0976356789469719]]], [[[0.10058089345693588]]], [[[0.5021613836288452]]], [[[0.7677112817764282]]], [[[0.9965066313743591]]]], dtype='float32').reshape([11, 1, 1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4d7dbe82a2b84077755612259360f4c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03ab246f0ef50d2f39d167b82cfcc661
    def get_inputs(self):
        return [
            paddle.uniform([11, 80, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0810799598693848], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_449996b1c13a28ef1a1e6fab04e4cf73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7d7822d1fed62091719529c0ea5eb7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d4ec0bb45ad0baf5f76c39400fc3d901(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9af9938a4076cf462823909dc39afd33
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b69b2a941125bd4fff9137864f4eef5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a145d1a0738c95ec3acc2366fecd6d2
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5f70510f7450b4c570c1e54d7d4cd0c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9600213b2b5203ea1935105a0554dbf7
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5311636ee7d398a450debdea9c5cbfc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.uniform([247], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7e74d5b9bf59624692482eacb60dfbd4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[152], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3d77bf48475521352623e55921847244(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e74d5b9bf59624692482eacb60dfbd4
    def get_inputs(self):
        return [
            paddle.uniform([152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_faaae291c89467569f79fb9332868a59(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[152], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_63a4f84aa01c6336bf25a941982c6d08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_faaae291c89467569f79fb9332868a59
    def get_inputs(self):
        return [
            paddle.uniform([152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_3b228569227788973e7076bc46119e26(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c90234efbbf024c357d560b9999dbd2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b228569227788973e7076bc46119e26
    def get_inputs(self):
        return [
            paddle.uniform([100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_5755eda1f828a71ac2bb3fa3a87ab770(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_85b70b091c477006d1ecf77aa485c07f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5755eda1f828a71ac2bb3fa3a87ab770
    def get_inputs(self):
        return [
            paddle.uniform([100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_1a9f6fa82df50ed4e89f4f8ed3c739a2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 152], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_472035a747dc0aa286c308c6ad5260c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a9f6fa82df50ed4e89f4f8ed3c739a2
    def get_inputs(self):
        return [
            paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_472035a747dc0aa286c308c6ad5260c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a9f6fa82df50ed4e89f4f8ed3c739a2
    def get_inputs(self):
        return [
            paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_37921c57166772a90456c548046e0e4c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 152], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_de84c06dd03b184c8f956147b7345d3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37921c57166772a90456c548046e0e4c
    def get_inputs(self):
        return [
            paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_de84c06dd03b184c8f956147b7345d3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37921c57166772a90456c548046e0e4c
    def get_inputs(self):
        return [
            paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_1b60c763cf6a1dcb445bf5b3ddba9d12(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[76], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bf4cd7f5288014b52f3ca0f47b7082b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b60c763cf6a1dcb445bf5b3ddba9d12
    def get_inputs(self):
        return [
            paddle.uniform([76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_b6a4243a8813d897a44e3638e49a9fad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[76], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_13b1c38022afbe81c155dafe889857bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6a4243a8813d897a44e3638e49a9fad
    def get_inputs(self):
        return [
            paddle.uniform([76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_ee6d740198e369f6d89e538d31cc4dac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[50], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e504945799d3c9a57e4e0d64dbf0601d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee6d740198e369f6d89e538d31cc4dac
    def get_inputs(self):
        return [
            paddle.uniform([50], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_fd919a2f39d9d308a1a49f82d5d128cd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[50], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_47d20b8c17fdcaa24b1fc7a7bdbdcdba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd919a2f39d9d308a1a49f82d5d128cd
    def get_inputs(self):
        return [
            paddle.uniform([50], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_8cd57fc586e62e400c03a3908704d831(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -64, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[50, 76], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f824fc370f36e0479117ac554f158d5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cd57fc586e62e400c03a3908704d831
    def get_inputs(self):
        return [
            paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f824fc370f36e0479117ac554f158d5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cd57fc586e62e400c03a3908704d831
    def get_inputs(self):
        return [
            paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_cabe26f84410859dbaddf0eac846908e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 64, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[50, 76], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e1f5eaf2e3f4df34afd8bad07177f586(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cabe26f84410859dbaddf0eac846908e
    def get_inputs(self):
        return [
            paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e1f5eaf2e3f4df34afd8bad07177f586(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cabe26f84410859dbaddf0eac846908e
    def get_inputs(self):
        return [
            paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_2eaac3fb0476c20293980a536c185087(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[38], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4c0a48e0f9a5ba1d5f24ae43c5541223(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2eaac3fb0476c20293980a536c185087
    def get_inputs(self):
        return [
            paddle.uniform([38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_ff07fc8d8a0a9e01b89def50cb63da97(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[38], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_775c09a20e32092ea363c8bad8a0ad08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff07fc8d8a0a9e01b89def50cb63da97
    def get_inputs(self):
        return [
            paddle.uniform([38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_f76b77f45738ca9ed91b978440476d48(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[25], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eff9195c8cb8507ce1a25037a0aa742f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f76b77f45738ca9ed91b978440476d48
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0], dtype='float32').reshape([25]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_530f73bb2a01b2c53db35654631b1ebc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[25], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7502e4ee916cf6e2ed836861acc7f35a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_530f73bb2a01b2c53db35654631b1ebc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5], dtype='float32').reshape([25]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_5743fb153583a117d383504af8e9af23(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -128, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[25, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_40f852d1052c56ef2fe88c569291795a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5743fb153583a117d383504af8e9af23
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_40f852d1052c56ef2fe88c569291795a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5743fb153583a117d383504af8e9af23
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c9b3bd8bf666a588ffb398384c3b379d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 128, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[25, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_97e2b1c7675640902943c34129586a58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9b3bd8bf666a588ffb398384c3b379d
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_97e2b1c7675640902943c34129586a58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9b3bd8bf666a588ffb398384c3b379d
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_85cd947ea62c1921e63a86db0ea7efb2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[19], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d7eabef8d485fa249a688046d1b7d8ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85cd947ea62c1921e63a86db0ea7efb2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0], dtype='float32').reshape([19]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_b30facf4c5afc2287577bec0afefed29(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[19], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e47ee3aa2343f9b9143d24ba01467e69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b30facf4c5afc2287577bec0afefed29
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5], dtype='float32').reshape([19]),
            paddle.to_tensor([64.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_dccd5293612feb652c792c547af6c471(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[13], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3b454d8c2b06a36cb78145cfc52e40d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dccd5293612feb652c792c547af6c471
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], dtype='float32').reshape([13]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c5bc633f1d9122996518e2c450bbffa1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[13], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_045439e8437fad5daa039824fd9d57e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5bc633f1d9122996518e2c450bbffa1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5], dtype='float32').reshape([13]),
            paddle.to_tensor([64.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_8212f75369683688e39183e264047c0b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -256, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[13, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_26b12c9e3ba2cb55eacad0e95402c9a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8212f75369683688e39183e264047c0b
    def get_inputs(self):
        return [
            paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_26b12c9e3ba2cb55eacad0e95402c9a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8212f75369683688e39183e264047c0b
    def get_inputs(self):
        return [
            paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c3836937c999df7482aa8e4d0675ecb9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 256, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[13, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ce8c3ac4af42e6d451f4ecb6f049a3b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3836937c999df7482aa8e4d0675ecb9
    def get_inputs(self):
        return [
            paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ce8c3ac4af42e6d451f4ecb6f049a3b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3836937c999df7482aa8e4d0675ecb9
    def get_inputs(self):
        return [
            paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_926342d73fb71f40207de21ef2cde0bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_497c6bda950bc99f455bfc0d819889a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_926342d73fb71f40207de21ef2cde0bf
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype='float32').reshape([10]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_498f07b19c75a46fdf5d2b261bbd510f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1124a273d03c19c2ac7e9d2d184e4726(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_498f07b19c75a46fdf5d2b261bbd510f
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5], dtype='float32').reshape([10]),
            paddle.to_tensor([128.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_97c344ff0d38ad7e4bd5cd96a431b4ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[7], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e69d13d8c5eee7c16b0b43bf78481320(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97c344ff0d38ad7e4bd5cd96a431b4ef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype='float32').reshape([7]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_602b9f17491ca3bc76dc0f385aaa87db(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[7], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_abf3195fad34f0023f9f70236d50be98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_602b9f17491ca3bc76dc0f385aaa87db
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], dtype='float32').reshape([7]),
            paddle.to_tensor([128.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_3ce8abfe8f28e5405912abb2476b8f95(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -512, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[7, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_af05aa9cd60d715b24778757b54dce82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ce8abfe8f28e5405912abb2476b8f95
    def get_inputs(self):
        return [
            paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_af05aa9cd60d715b24778757b54dce82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ce8abfe8f28e5405912abb2476b8f95
    def get_inputs(self):
        return [
            paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_a5f3a8932b4f4ab853c6d119c81dff77(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 512, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[7, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0218acb562ca31dbe9741bb1efebece3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5f3a8932b4f4ab853c6d119c81dff77
    def get_inputs(self):
        return [
            paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0218acb562ca31dbe9741bb1efebece3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5f3a8932b4f4ab853c6d119c81dff77
    def get_inputs(self):
        return [
            paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_6cace5d5abfae5e10d7bb959e485d915(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_957392cba0ec1940cbaf467412bb652a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6cace5d5abfae5e10d7bb959e485d915
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_40e638a545923bbd854852fb93954c00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d4dc2965698bbfff26e52d07448b60b
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a7092c3093f63acd441079bee597e041(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b2435231cb0b72f5bf9fbe721fbabae
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([20]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_19566edb7c5c47fab98f3b44e39958bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.to_tensor([2.0487473011016846, 2.2904977798461914, 2.0161566734313965, 2.1081438064575195, 2.213404893875122, 2.0244739055633545, 1.9477956295013428, 2.2595083713531494, 1.9438437223434448, 1.9874775409698486, 2.1578941345214844, 2.097313404083252, 2.1363720893859863, 2.163761615753174, 2.058284044265747, 2.046527147293091, 2.133448362350464, 2.0699000358581543, 2.12774395942688, 2.0742523670196533], dtype='float32').reshape([20]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8a48f1f821920fbad20ea7f488a58744(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(2.55885648727417, dtype='float32').reshape([]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_27e3cc44a370bfc6c5cd9ce623c689ce(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 4096, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_df0bf7534f4a39a1f7d8b061ccaee5d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27e3cc44a370bfc6c5cd9ce623c689ce
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e5d757abcffe63788b0ef822bbf1c62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27e3cc44a370bfc6c5cd9ce623c689ce
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_97f9d1937249a96b510c38034ed3bb7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4526b35f2d10614b12d93f6b96e2ec89
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_52d4dd59305a9f8794d7b34f7542cc33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_97f9d1937249a96b510c38034ed3bb7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4526b35f2d10614b12d93f6b96e2ec89
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_52d4dd59305a9f8794d7b34f7542cc33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b2342d6bed12006e2a9f4bb467835265(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4526b35f2d10614b12d93f6b96e2ec89
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ade2bcd145584cdcb79b6c9762a29444(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b2342d6bed12006e2a9f4bb467835265(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4526b35f2d10614b12d93f6b96e2ec89
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ade2bcd145584cdcb79b6c9762a29444(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8c436de5f23112c51a72421541a4cae6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4526b35f2d10614b12d93f6b96e2ec89
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_083b4d3efaf3cf209b1d805420a3eb94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8c436de5f23112c51a72421541a4cae6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4526b35f2d10614b12d93f6b96e2ec89
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_083b4d3efaf3cf209b1d805420a3eb94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e3493d17b8af44deb135729d4f75a17b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(364.0583190917969, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f6cc3ffd20a1566be5d122edeeefa29f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a145d1a0738c95ec3acc2366fecd6d2
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_aa6e9403734cc36380a9d6e212b570e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1811779b2a5c416b22e37c688b06c8c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.014916401356458664], [0.027211245149374008], [0.014795346185564995], [0.052663881331682205]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1a0462561e7926b3db4d7a86fef600d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1811779b2a5c416b22e37c688b06c8c
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.03851116821169853], [0.19017067551612854], [0.06972962617874146], [0.15650993585586548]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6fff48549287e1e1c06e1780f81523c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1255f7a4bcebf297eee2ebc8e49a136
    def get_inputs(self):
        return [
            paddle.to_tensor([[-1.3873265981674194], [-0.8569114208221436], [-0.7878183126449585], [-0.6635109782218933]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4901637a65ff84e1a675a40a99169c6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b4e3ffb4da18304ddf9f1e56baa2416
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.387326717376709], [1.8569114208221436], [1.7878183126449585], [1.663511037826538]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_027d47ab7388419df1e35b3e27463a52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.uniform([70], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_56860b453692e32b713803f7346c1c74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(32.68608093261719, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_648d79f97cba0397cdf9583a0d8e4753(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b9c3e62b05bfc6cd607c3af196c561e
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f902490c41039fc84df3624c2f0eb678(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03ab246f0ef50d2f39d167b82cfcc661
    def get_inputs(self):
        return [
            paddle.uniform([43, 40, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0526299476623535], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bd19e7322f82839b9a05b1c6b627648d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95864ebc6da496802f63c6f82174b6ba
    def get_inputs(self):
        return [
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bd19e7322f82839b9a05b1c6b627648d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95864ebc6da496802f63c6f82174b6ba
    def get_inputs(self):
        return [
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_dd75ccf870a5f9c457937202607ed82b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bce202c470bbbf07a610c93dd97477d6
    def get_inputs(self):
        return [
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5e1a381b790e2372673da9a3aea8c7f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_52c1c16ac8d019b900f1c735a7d7334d
    def get_inputs(self):
        return [
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f3de272e76c1094e2c344c07d5c1da97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86299bdcdf08bf954fc6518b813b5b7f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2092, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4dd7473860d6d4a6c86c48184298b3be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_626de8f444c822bb894fb31988f051d3
    def get_inputs(self):
        return [
            paddle.uniform([2092, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f2759664b11e501d0614309b80d07ccd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71bd1f02664b7f7df6bc8e3f5c3bb387
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2092, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f2759664b11e501d0614309b80d07ccd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71bd1f02664b7f7df6bc8e3f5c3bb387
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2092, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ec2905907300b200db91e8cb3918d621(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06f90143b8ab9d69120ff1073a9dc131
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_56a5f2db30fd4b0714a8deff75ca2a6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_066290e400a309599edf8c14cfc4a104
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0e4788abf81964905d04a1c809a07aff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b0be87a58810bed51f2f3b6995e9ef2
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_65e62d76aa6ab5bfd43e8e94ed365440(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccf84142974138841b80fa5a1f911d0b
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 197, 197], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_00f12038eff5bacca8f01b1461e89088(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07da1b56c4289e55404bebb46f57a9ec
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cd974b6e350999fcd9c1adec4a01251b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5979570d19df229baf6bb203b1dfcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1e784cf8f1a282baffbfa44b5241b877(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d5f9bd71fabbc72e3807e6e4c080102
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cd974b6e350999fcd9c1adec4a01251b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5979570d19df229baf6bb203b1dfcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5b2beaaff5d49b678e4d0044967b0344(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9600213b2b5203ea1935105a0554dbf7
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0b8193bb6433511d16a561cf11d8367f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.uniform([551], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_9967ca72578dd8ef3e14c36b7f9d35f1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1e42b9f37300328bc546260cad36311f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9967ca72578dd8ef3e14c36b7f9d35f1
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_13b972edb47b6d983dc2f354bf4fca05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9967ca72578dd8ef3e14c36b7f9d35f1
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_6757a10f49dd3c5dedfad8c4ebb646bc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_73ebf17bea752066c844306b0bf2d61f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6757a10f49dd3c5dedfad8c4ebb646bc
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b524bd9d1a6b8fbd43c50887393c3dc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6757a10f49dd3c5dedfad8c4ebb646bc
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_5fba441d1e8e41bfd01c2535838fdf58(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_209f7ce44af0acb2c5b01e3c33e5bb1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fba441d1e8e41bfd01c2535838fdf58
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0], dtype='float32').reshape([18]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_58bbdf71533ebb929305aa568288edcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fba441d1e8e41bfd01c2535838fdf58
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0], dtype='float32').reshape([18]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e422b9fa52ba59149e6ae38ba5277ffe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9908f0f4dbc186c67d7a4948572047
    def get_inputs(self):
        return [
            paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7777138a7c1729c3de2464c2da242473(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5d2f11747180d586348616f938df2b9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7777138a7c1729c3de2464c2da242473
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_13b972edb47b6d983dc2f354bf4fca05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9967ca72578dd8ef3e14c36b7f9d35f1
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_64b358990ae4cc63a7581fb78595c31d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4bce05b28be8849111aadbe1a45b6c8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64b358990ae4cc63a7581fb78595c31d
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b524bd9d1a6b8fbd43c50887393c3dc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6757a10f49dd3c5dedfad8c4ebb646bc
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_1d30cf08f36479f751b3b34565d68667(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bc764c5c16a091f84fe028f98b70c5b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d30cf08f36479f751b3b34565d68667
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0], dtype='float32').reshape([18]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_355a82ae40b18f1e6f9ebfbb89d31790(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fba441d1e8e41bfd01c2535838fdf58
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5], dtype='float32').reshape([18]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7545102ca6a6bfa73320882a80a580ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(135.29037475585938, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_be4ee26d0a4a323a37a2f9ea5e758449(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(2.018048048019409, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1cf814b543620e69c96663c8ed99f745(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7d7822d1fed62091719529c0ea5eb7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_48a8a2869d47ae2d59b077d900ed3ac3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0884743459f4915d82904c5015982b8f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5f72d3a7c9eeed9fa2f58c2e1e63c270(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9af9938a4076cf462823909dc39afd33
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_9d5ad4d27e5434eb5bc012aa5880a256(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, None, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_35e8ee17faf13989d562d6f47bb1bf89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d5ad4d27e5434eb5bc012aa5880a256
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 160, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5dead5fafb19f294ddc54ffa5e299325(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e4922968922a1a6756c98fbf770d107
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1174, 1174], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_77663c9ececcd0cf1207f3f9817844a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.uniform([3800], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3d96451fa931d16b878746c7ee8f7e98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b0be87a58810bed51f2f3b6995e9ef2
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1ce50f6c2e2b76acad44c8afab79b274(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.uniform([2204], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ceed4979b7bffd85870a207cc3ec8e61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03ab246f0ef50d2f39d167b82cfcc661
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_aaf9eb01860bac2d07fefe1b5ccbb9f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7d7822d1fed62091719529c0ea5eb7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e05a45b5dabc87aec71a5d78abb25a5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0884743459f4915d82904c5015982b8f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e8be4fab207da0540c569eae7a0981f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9af9938a4076cf462823909dc39afd33
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_15960a13f628bd8e7ab53884f1de03ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(36.610042572021484, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_24971a95fd7ed8e62effa7fc3ccbafd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7d7822d1fed62091719529c0ea5eb7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b28c88138bdc7a3e8bb4216d76f6e4d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9af9938a4076cf462823909dc39afd33
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0ea5bb41c3be9de01d2d360234770b5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95864ebc6da496802f63c6f82174b6ba
    def get_inputs(self):
        return [
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0ea5bb41c3be9de01d2d360234770b5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95864ebc6da496802f63c6f82174b6ba
    def get_inputs(self):
        return [
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_462b0270075033280739664ce9b8f9d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bce202c470bbbf07a610c93dd97477d6
    def get_inputs(self):
        return [
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e4a8b0b9062921e6332c417a3a2e302b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_52c1c16ac8d019b900f1c735a7d7334d
    def get_inputs(self):
        return [
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_67de0eba3fbc18c0cbd8334101ce1ff0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86299bdcdf08bf954fc6518b813b5b7f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4214, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_412b40cf64da004ea74ff7db48c92496(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_626de8f444c822bb894fb31988f051d3
    def get_inputs(self):
        return [
            paddle.uniform([4214, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_305d3ac97da8e6b912d0e7631509c3e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71bd1f02664b7f7df6bc8e3f5c3bb387
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4214, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_305d3ac97da8e6b912d0e7631509c3e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71bd1f02664b7f7df6bc8e3f5c3bb387
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4214, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_02e6832d100596eadba4367a480c72cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b6147eb9e3d4540fa230991620e9845
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1174, 1174], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_93104c38c3091acd9b13b312e1300936(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aea8ae797393ab8e346cadca33845a8b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_60a9db8419058fb838738ecb7f984a08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(101.66637420654297, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d78faf745cd44417515695b702934cde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(4.152548789978027, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_97f9d1937249a96b510c38034ed3bb7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4526b35f2d10614b12d93f6b96e2ec89
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_52d4dd59305a9f8794d7b34f7542cc33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_97f9d1937249a96b510c38034ed3bb7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4526b35f2d10614b12d93f6b96e2ec89
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_52d4dd59305a9f8794d7b34f7542cc33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b2342d6bed12006e2a9f4bb467835265(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4526b35f2d10614b12d93f6b96e2ec89
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ade2bcd145584cdcb79b6c9762a29444(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b2342d6bed12006e2a9f4bb467835265(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4526b35f2d10614b12d93f6b96e2ec89
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ade2bcd145584cdcb79b6c9762a29444(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8c436de5f23112c51a72421541a4cae6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4526b35f2d10614b12d93f6b96e2ec89
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_083b4d3efaf3cf209b1d805420a3eb94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8c436de5f23112c51a72421541a4cae6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4526b35f2d10614b12d93f6b96e2ec89
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_083b4d3efaf3cf209b1d805420a3eb94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e64fa8c8e1a2e2bcfa7741a33ad3ec
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_706ff3d631b9c5268eed10b41b3142ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d5ad4d27e5434eb5bc012aa5880a256
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 50, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_24971a95fd7ed8e62effa7fc3ccbafd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7d7822d1fed62091719529c0ea5eb7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ad1e091cd24dceeb0a3930800227a83f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0884743459f4915d82904c5015982b8f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_295b1f166c8ce3f95b87de38ce76daa1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9af9938a4076cf462823909dc39afd33
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9b585574ec71fbc2d5c925ebeb4c7adb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699166084feb39f20e7855e1829a3ad4
    def get_inputs(self):
        return [
            paddle.to_tensor(29.75494384765625, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()