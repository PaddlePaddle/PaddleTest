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



class PrimitiveOp_b486fb2cfac6a4b9624be2eab74dce91(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2986c2684c72fbb12868b9990399e81f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b486fb2cfac6a4b9624be2eab74dce91
    def get_inputs(self):
        return [
            paddle.uniform([300, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[300, 1], dtype='int32'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_f3da6fa3c4c6ee028c4f3fb559cdcf07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b486fb2cfac6a4b9624be2eab74dce91
    def get_inputs(self):
        return [
            paddle.uniform([8, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6], [7]], dtype='int32').reshape([8, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_c78dae6038ac8adda0d925f7e1625d95(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3471c864738cfc15a6f7a048a2ee453b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c78dae6038ac8adda0d925f7e1625d95
    def get_inputs(self):
        return [
            paddle.uniform([2, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [1]], dtype='int32').reshape([2, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_f2ddd3965e7de4530094fb3e94ca3e6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b486fb2cfac6a4b9624be2eab74dce91
    def get_inputs(self):
        return [
            paddle.uniform([100, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[100, 1], dtype='int32'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_f4ad3cd62341f5d1a81d825589be2520(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5ab847d7e51eb3bbf9c4c379c5ca8c2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4ad3cd62341f5d1a81d825589be2520
    def get_inputs(self):
        return [
            paddle.to_tensor([3], dtype='int32').reshape([1]),
            paddle.randint(low=0, high=3, shape=[2100], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_3ce2b180e87cf18107ae0a4505638517(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_05ecb4339abb2eae24510bd4aaee2a97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ce2b180e87cf18107ae0a4505638517
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.00428850669413805, 0.3288317620754242, 0.4979742765426636, 0.23877312242984772]], dtype='float32').reshape([1, 4]),
            paddle.randint(low=0, high=3, shape=[2100], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_27034c450c424231888ce0bdf9c27fbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b486fb2cfac6a4b9624be2eab74dce91
    def get_inputs(self):
        return [
            paddle.uniform([2, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [1]], dtype='int32').reshape([2, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_866d0b647e17768cf8900ace3c007516(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_12ffd449822f0de9022ed4098c393dcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_866d0b647e17768cf8900ace3c007516
    def get_inputs(self):
        return [
            paddle.uniform([185691], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_6df22063311e4421bbd3d492da35aca0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2c20c5c468413faa0f563adb038c6bfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6df22063311e4421bbd3d492da35aca0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[185691], dtype='int32'),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_df5062ca967442de395dfb80f2285f9b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c5be1b35cd5e5acd57d5ff070aa1f025(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df5062ca967442de395dfb80f2285f9b
    def get_inputs(self):
        return [
            paddle.uniform([185691, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1], [1], [5], [0], [9], [2], [4], [2]], dtype='int64').reshape([8, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5be1b35cd5e5acd57d5ff070aa1f025(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df5062ca967442de395dfb80f2285f9b
    def get_inputs(self):
        return [
            paddle.uniform([185691, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1], [1], [5], [0], [9], [2], [4], [2]], dtype='int64').reshape([8, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_3471c864738cfc15a6f7a048a2ee453b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c78dae6038ac8adda0d925f7e1625d95
    def get_inputs(self):
        return [
            paddle.uniform([2, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [1]], dtype='int32').reshape([2, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_f3da6fa3c4c6ee028c4f3fb559cdcf07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b486fb2cfac6a4b9624be2eab74dce91
    def get_inputs(self):
        return [
            paddle.uniform([8, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6], [7]], dtype='int32').reshape([8, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_0d51659c5a185ad2b9e8b29e62c858c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4ad3cd62341f5d1a81d825589be2520
    def get_inputs(self):
        return [
            paddle.to_tensor([9, 5], dtype='int32').reshape([2]),
            paddle.randint(low=0, high=3, shape=[2002], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_2da2d3aad92626ad3cdd2e026a0396b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4ad3cd62341f5d1a81d825589be2520
    def get_inputs(self):
        return [
            paddle.to_tensor([6, 0, 2, 8, 9, 6, 2, 5, 4, 0, 2, 4, 2, 2, 3, 5, 2, 4, 4, 1, 0], dtype='int32').reshape([21]),
            paddle.randint(low=0, high=3, shape=[1021], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_f39ac59bff1d2316658e5b25372e33f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_866d0b647e17768cf8900ace3c007516
    def get_inputs(self):
        return [
            paddle.uniform([242991], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_13d9a8acb9e44fe94854668a266dca0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6df22063311e4421bbd3d492da35aca0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[242991], dtype='int32'),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_a659e360e24f18abbe2c646388a89bd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df5062ca967442de395dfb80f2285f9b
    def get_inputs(self):
        return [
            paddle.uniform([242991, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[8], [0], [6], [1], [5]], dtype='int64').reshape([5, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_a659e360e24f18abbe2c646388a89bd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df5062ca967442de395dfb80f2285f9b
    def get_inputs(self):
        return [
            paddle.uniform([242991, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[8], [0], [6], [1], [5]], dtype='int64').reshape([5, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_f2c5160af211b2ed20424181b8b20f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b486fb2cfac6a4b9624be2eab74dce91
    def get_inputs(self):
        return [
            paddle.uniform([7, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6]], dtype='int32').reshape([7, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b38e3f8c5ae1fc8a1fb28891fe31aa3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4ad3cd62341f5d1a81d825589be2520
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 5], dtype='int32').reshape([2]),
            paddle.randint(low=0, high=3, shape=[1002], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_e87ee50c6bd8b7e3f83cb859737716a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_866d0b647e17768cf8900ace3c007516
    def get_inputs(self):
        return [
            paddle.uniform([171888], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_fc34a5dd236ca4f649d3d00cd755041c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6df22063311e4421bbd3d492da35aca0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[171888], dtype='int32'),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8745e00c995132731010b61f6475f053(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df5062ca967442de395dfb80f2285f9b
    def get_inputs(self):
        return [
            paddle.uniform([171888, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[6], [4], [1], [4], [1]], dtype='int64').reshape([5, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8745e00c995132731010b61f6475f053(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df5062ca967442de395dfb80f2285f9b
    def get_inputs(self):
        return [
            paddle.uniform([171888, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[6], [4], [1], [4], [1]], dtype='int64').reshape([5, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_8f3c4fc99580f866c5e3798ec6571465(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_507fcbd6bbc3e03f6bab560b59e9ab97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f3c4fc99580f866c5e3798ec6571465
    def get_inputs(self):
        return [
            paddle.uniform([6, 256, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [1], [2], [3], [4], [5]], dtype='int32').reshape([6, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_e87ee50c6bd8b7e3f83cb859737716a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_866d0b647e17768cf8900ace3c007516
    def get_inputs(self):
        return [
            paddle.uniform([171888], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_fc34a5dd236ca4f649d3d00cd755041c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6df22063311e4421bbd3d492da35aca0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[171888], dtype='int32'),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_755a8d0b1cb96db2101f18f6e3520f74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df5062ca967442de395dfb80f2285f9b
    def get_inputs(self):
        return [
            paddle.uniform([171888, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[6], [4], [1], [4], [1], [3], [3]], dtype='int64').reshape([7, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_755a8d0b1cb96db2101f18f6e3520f74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df5062ca967442de395dfb80f2285f9b
    def get_inputs(self):
        return [
            paddle.uniform([171888, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[6], [4], [1], [4], [1], [3], [3]], dtype='int64').reshape([7, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_a2a76d9721a6d9f733526aa594a5472a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b486fb2cfac6a4b9624be2eab74dce91
    def get_inputs(self):
        return [
            paddle.uniform([3, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [1], [2]], dtype='int32').reshape([3, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_f0480238b941cd6d0a1705d7cd77196c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_866d0b647e17768cf8900ace3c007516
    def get_inputs(self):
        return [
            paddle.uniform([217413], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_487ce49c437a6aa75bd990ad1170a823(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6df22063311e4421bbd3d492da35aca0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[217413], dtype='int32'),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_ee8811d0501287303d8b5f2dbf8b2587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df5062ca967442de395dfb80f2285f9b
    def get_inputs(self):
        return [
            paddle.uniform([217413, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[103, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_ee8811d0501287303d8b5f2dbf8b2587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df5062ca967442de395dfb80f2285f9b
    def get_inputs(self):
        return [
            paddle.uniform([217413, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[103, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_3471c864738cfc15a6f7a048a2ee453b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c78dae6038ac8adda0d925f7e1625d95
    def get_inputs(self):
        return [
            paddle.uniform([2, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [1]], dtype='int32').reshape([2, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d370067ecbf584b6a26d0c21fb4e61bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f3c4fc99580f866c5e3798ec6571465
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0]], dtype='int32').reshape([1, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_487020cd1656f85aa7353097272ee4db(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_be7b1fcb4a53674abb9607cb73a4cdef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4ad3cd62341f5d1a81d825589be2520
    def get_inputs(self):
        return [
            paddle.to_tensor([6, 6], dtype='int32').reshape([2]),
            paddle.randint(low=0, high=3, shape=[3549], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_1583d7812ffaef688ec673af20704d7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ce2b180e87cf18107ae0a4505638517
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3754444122314453, 0.4557427167892456, 0.2475605010986328, 0.39700043201446533], [0.41998299956321716, 0.3809603452682495, 0.07890275120735168, 0.16198381781578064]], dtype='float32').reshape([2, 4]),
            paddle.randint(low=0, high=3, shape=[3549], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_7392f5f1fd506b46a9739bb3d60f8c10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b486fb2cfac6a4b9624be2eab74dce91
    def get_inputs(self):
        return [
            paddle.uniform([7, 64, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6]], dtype='int32').reshape([7, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d370067ecbf584b6a26d0c21fb4e61bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f3c4fc99580f866c5e3798ec6571465
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0]], dtype='int32').reshape([1, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_aa0d853f47a278fe768e6cdbf98a0ca0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_866d0b647e17768cf8900ace3c007516
    def get_inputs(self):
        return [
            paddle.uniform([86970], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_0ad4bd07d1f905bd29a18316b776248b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6df22063311e4421bbd3d492da35aca0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[86970], dtype='int32'),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6f6065fb0c27efcb5a6fe2e096c4e01d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df5062ca967442de395dfb80f2285f9b
    def get_inputs(self):
        return [
            paddle.uniform([86970, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[9], [5], [1], [0], [0], [1]], dtype='int64').reshape([6, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6f6065fb0c27efcb5a6fe2e096c4e01d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df5062ca967442de395dfb80f2285f9b
    def get_inputs(self):
        return [
            paddle.uniform([86970, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[9], [5], [1], [0], [0], [1]], dtype='int64').reshape([6, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_125de1469c91d1717f47f8ab4b8188a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_866d0b647e17768cf8900ace3c007516
    def get_inputs(self):
        return [
            paddle.uniform([205923], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6f48f5be9f4583a6bebaf965d86146e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6df22063311e4421bbd3d492da35aca0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[205923], dtype='int32'),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_12a4646d3c81a5529f25ec0c48ad49d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df5062ca967442de395dfb80f2285f9b
    def get_inputs(self):
        return [
            paddle.uniform([205923, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1], [0], [8], [4], [1]], dtype='int64').reshape([5, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_12a4646d3c81a5529f25ec0c48ad49d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df5062ca967442de395dfb80f2285f9b
    def get_inputs(self):
        return [
            paddle.uniform([205923, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1], [0], [8], [4], [1]], dtype='int64').reshape([5, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_66b3f3cf17b00484ca76018d2965888a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_866d0b647e17768cf8900ace3c007516
    def get_inputs(self):
        return [
            paddle.uniform([153450], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_71966051d2f853d8d8a16f9aec598de1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6df22063311e4421bbd3d492da35aca0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[153450], dtype='int32'),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_7e3454cefdc16aad370fb8c17c0a365e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df5062ca967442de395dfb80f2285f9b
    def get_inputs(self):
        return [
            paddle.uniform([153450, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[8], [4], [4], [2], [3], [1], [7], [4], [8], [3]], dtype='int64').reshape([10, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_7e3454cefdc16aad370fb8c17c0a365e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df5062ca967442de395dfb80f2285f9b
    def get_inputs(self):
        return [
            paddle.uniform([153450, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[8], [4], [4], [2], [3], [1], [7], [4], [8], [3]], dtype='int64').reshape([10, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_ed0ebd4973cfdabcd03bf95ddc3eec4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b486fb2cfac6a4b9624be2eab74dce91
    def get_inputs(self):
        return [
            paddle.uniform([5, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [1], [2], [3], [4]], dtype='int32').reshape([5, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_4f8249b76bd8fc5850e63cc4bc9818de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4ad3cd62341f5d1a81d825589be2520
    def get_inputs(self):
        return [
            paddle.to_tensor([3], dtype='int32').reshape([1]),
            paddle.randint(low=0, high=3, shape=[4116], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_666cfe71908f34ec069541450a44de67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ce2b180e87cf18107ae0a4505638517
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.12708094716072083, 0.308269739151001, 0.40365836024284363, 0.38234201073646545]], dtype='float32').reshape([1, 4]),
            paddle.randint(low=0, high=3, shape=[4116], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_f2c5160af211b2ed20424181b8b20f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b486fb2cfac6a4b9624be2eab74dce91
    def get_inputs(self):
        return [
            paddle.uniform([7, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6]], dtype='int32').reshape([7, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_e9be5315292a1b4d5ca6eb618f84e5b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_866d0b647e17768cf8900ace3c007516
    def get_inputs(self):
        return [
            paddle.uniform([113061], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_e33749830d460e132ef51edaa3894f01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6df22063311e4421bbd3d492da35aca0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[113061], dtype='int32'),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_93422d47bb4f84ff1e959337728dbb07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df5062ca967442de395dfb80f2285f9b
    def get_inputs(self):
        return [
            paddle.uniform([113061, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[2], [6], [7], [8]], dtype='int64').reshape([4, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_93422d47bb4f84ff1e959337728dbb07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df5062ca967442de395dfb80f2285f9b
    def get_inputs(self):
        return [
            paddle.uniform([113061, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[2], [6], [7], [8]], dtype='int64').reshape([4, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_f2c5160af211b2ed20424181b8b20f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b486fb2cfac6a4b9624be2eab74dce91
    def get_inputs(self):
        return [
            paddle.uniform([7, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6]], dtype='int32').reshape([7, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d370067ecbf584b6a26d0c21fb4e61bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f3c4fc99580f866c5e3798ec6571465
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0]], dtype='int32').reshape([1, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_f70182eed5a43530a26de7ee50e1a8f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_866d0b647e17768cf8900ace3c007516
    def get_inputs(self):
        return [
            paddle.uniform([123783], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_80bbba188ddedd15b95b20543fbff7c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6df22063311e4421bbd3d492da35aca0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[123783], dtype='int32'),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_88d9a094ab81a7ebfb450360d5bc42ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df5062ca967442de395dfb80f2285f9b
    def get_inputs(self):
        return [
            paddle.uniform([123783, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[84, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_88d9a094ab81a7ebfb450360d5bc42ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df5062ca967442de395dfb80f2285f9b
    def get_inputs(self):
        return [
            paddle.uniform([123783, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[84, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_2986c2684c72fbb12868b9990399e81f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b486fb2cfac6a4b9624be2eab74dce91
    def get_inputs(self):
        return [
            paddle.uniform([300, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[300, 1], dtype='int32'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_12ffd449822f0de9022ed4098c393dcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_866d0b647e17768cf8900ace3c007516
    def get_inputs(self):
        return [
            paddle.uniform([185691], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_2c20c5c468413faa0f563adb038c6bfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6df22063311e4421bbd3d492da35aca0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[185691], dtype='int32'),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c12398248138a696ff1e717cd9203b29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df5062ca967442de395dfb80f2285f9b
    def get_inputs(self):
        return [
            paddle.uniform([185691, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1], [1], [5], [0], [9], [2], [4]], dtype='int64').reshape([7, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c12398248138a696ff1e717cd9203b29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df5062ca967442de395dfb80f2285f9b
    def get_inputs(self):
        return [
            paddle.uniform([185691, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1], [1], [5], [0], [9], [2], [4]], dtype='int64').reshape([7, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_ed0ebd4973cfdabcd03bf95ddc3eec4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b486fb2cfac6a4b9624be2eab74dce91
    def get_inputs(self):
        return [
            paddle.uniform([5, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [1], [2], [3], [4]], dtype='int32').reshape([5, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_66b3f3cf17b00484ca76018d2965888a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_866d0b647e17768cf8900ace3c007516
    def get_inputs(self):
        return [
            paddle.uniform([153450], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_71966051d2f853d8d8a16f9aec598de1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6df22063311e4421bbd3d492da35aca0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[153450], dtype='int32'),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c7024ec93286c9e58bf353e54c6ece2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df5062ca967442de395dfb80f2285f9b
    def get_inputs(self):
        return [
            paddle.uniform([153450, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[8], [4], [4], [2], [3], [1]], dtype='int64').reshape([6, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c7024ec93286c9e58bf353e54c6ece2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df5062ca967442de395dfb80f2285f9b
    def get_inputs(self):
        return [
            paddle.uniform([153450, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[8], [4], [4], [2], [3], [1]], dtype='int64').reshape([6, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_2bafddcf3aea558fc1ecafc96ba70b38(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_87bba74fe34258c83835c6e43289c635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bafddcf3aea558fc1ecafc96ba70b38
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_87bba74fe34258c83835c6e43289c635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bafddcf3aea558fc1ecafc96ba70b38
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_87bba74fe34258c83835c6e43289c635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bafddcf3aea558fc1ecafc96ba70b38
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_87bba74fe34258c83835c6e43289c635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bafddcf3aea558fc1ecafc96ba70b38
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_87bba74fe34258c83835c6e43289c635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bafddcf3aea558fc1ecafc96ba70b38
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_87bba74fe34258c83835c6e43289c635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bafddcf3aea558fc1ecafc96ba70b38
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_87bba74fe34258c83835c6e43289c635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bafddcf3aea558fc1ecafc96ba70b38
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_87bba74fe34258c83835c6e43289c635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bafddcf3aea558fc1ecafc96ba70b38
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_87bba74fe34258c83835c6e43289c635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bafddcf3aea558fc1ecafc96ba70b38
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_87bba74fe34258c83835c6e43289c635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bafddcf3aea558fc1ecafc96ba70b38
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_87bba74fe34258c83835c6e43289c635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bafddcf3aea558fc1ecafc96ba70b38
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_87bba74fe34258c83835c6e43289c635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bafddcf3aea558fc1ecafc96ba70b38
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_87bba74fe34258c83835c6e43289c635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bafddcf3aea558fc1ecafc96ba70b38
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_87bba74fe34258c83835c6e43289c635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bafddcf3aea558fc1ecafc96ba70b38
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_87bba74fe34258c83835c6e43289c635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bafddcf3aea558fc1ecafc96ba70b38
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_87bba74fe34258c83835c6e43289c635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bafddcf3aea558fc1ecafc96ba70b38
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_579b0663729b4b5ed6f510777c43ad65(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dbd4e97d752c77c16c5d989a8d0bde0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_579b0663729b4b5ed6f510777c43ad65
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 3], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dbd4e97d752c77c16c5d989a8d0bde0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_579b0663729b4b5ed6f510777c43ad65
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 3], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8acef5f33816a6e40ed2b7be40d1dcf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_579b0663729b4b5ed6f510777c43ad65
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 9], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8acef5f33816a6e40ed2b7be40d1dcf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_579b0663729b4b5ed6f510777c43ad65
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 9], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c32d896c25094a7e923507f0bccae7a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b486fb2cfac6a4b9624be2eab74dce91
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0]], dtype='int32').reshape([1, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d370067ecbf584b6a26d0c21fb4e61bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f3c4fc99580f866c5e3798ec6571465
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0]], dtype='int32').reshape([1, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_ac83316343497f29f765535ab797a74e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4ad3cd62341f5d1a81d825589be2520
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 2, 3, 5, 2, 4, 4, 1, 0, 6, 8, 6, 0, 6, 9, 3, 4, 9, 4, 0, 0, 7, 8, 6, 1, 9, 3], dtype='int32').reshape([27]),
            paddle.randint(low=0, high=3, shape=[1027], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_5f7d00927b3c9aaab6f92fa8ada7e3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f3c4fc99580f866c5e3798ec6571465
    def get_inputs(self):
        return [
            paddle.uniform([8, 256, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6], [7]], dtype='int32').reshape([8, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_f2ddd3965e7de4530094fb3e94ca3e6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b486fb2cfac6a4b9624be2eab74dce91
    def get_inputs(self):
        return [
            paddle.uniform([100, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[100, 1], dtype='int32'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d370067ecbf584b6a26d0c21fb4e61bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f3c4fc99580f866c5e3798ec6571465
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0]], dtype='int32').reshape([1, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_3228fe74744759722574e864e7f8e3bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_866d0b647e17768cf8900ace3c007516
    def get_inputs(self):
        return [
            paddle.uniform([220968], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_e15d8ecdd81574617f1db4fa0e6cb0de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6df22063311e4421bbd3d492da35aca0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[220968], dtype='int32'),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_a2a7e986e0e9d1aaf74e4a046c111612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df5062ca967442de395dfb80f2285f9b
    def get_inputs(self):
        return [
            paddle.uniform([220968, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[6], [5], [2], [2], [8]], dtype='int64').reshape([5, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_a2a7e986e0e9d1aaf74e4a046c111612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df5062ca967442de395dfb80f2285f9b
    def get_inputs(self):
        return [
            paddle.uniform([220968, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[6], [5], [2], [2], [8]], dtype='int64').reshape([5, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_edc422e3538b0c872610add76650571e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_fe23d8d061a6333391c64fdc7b2deeb7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_10c8b2a3ce7a36e75a7f9375713c6242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe23d8d061a6333391c64fdc7b2deeb7
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 2, 0, 0, 2, 1, 0, 0, 1, 0, 2, 2, 2, 0, 1, 2], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_bf2a1246d6a8da3d4157b8c385638d70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe23d8d061a6333391c64fdc7b2deeb7
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 2, 1, 1, 2, 1, 1, 0, 0, 1, 1, 2, 1, 1, 2, 2], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8b01429a66347e3b4f78d935d0b53148(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe23d8d061a6333391c64fdc7b2deeb7
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1, 0, 1, 2, 1, 0, 0, 1, 0, 2, 2, 2, 2, 2, 1], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_5ad7cc1e866f1cbd65d6c582953013c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe23d8d061a6333391c64fdc7b2deeb7
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 0, 2, 1, 1, 1, 2, 1, 2, 1, 0, 0, 0, 2, 2, 1], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6eb8a9e2c2e13d43138d311c9612a7cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe23d8d061a6333391c64fdc7b2deeb7
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 2, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 2, 0, 1, 1], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dcd81e50c7c09444f806e67518db2c3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe23d8d061a6333391c64fdc7b2deeb7
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1, 1, 2, 2, 2, 1, 0, 1, 0, 0, 0, 2, 2, 1, 1], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dcff8b01367055c935c22869f63048fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe23d8d061a6333391c64fdc7b2deeb7
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 2, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 2, 0, 0, 2], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_a96382bc78ab65e36d0193e4723fb098(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe23d8d061a6333391c64fdc7b2deeb7
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1, 1, 0, 1, 2, 0, 0, 2, 1, 2, 1, 0, 0, 2, 0], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_562186042759adb831e25576ad56313e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe23d8d061a6333391c64fdc7b2deeb7
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 0, 1, 1, 1, 1, 1, 0, 2, 1, 0, 0, 2, 2, 1, 1], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_708abb0809892882db77cd7a1ec12acc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe23d8d061a6333391c64fdc7b2deeb7
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 2, 0, 1, 0, 0, 1, 1, 2, 2, 0, 2, 1, 1, 2, 1], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_7772820f68fc3f45dc45bbbbbaaa95c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe23d8d061a6333391c64fdc7b2deeb7
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 1, 2, 0, 0, 1, 2, 0, 2, 2, 2, 1, 1, 0, 0, 0], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8965ce895f9d2f5e41a41f1545196af4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe23d8d061a6333391c64fdc7b2deeb7
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1, 0, 2, 2, 1, 2, 0, 2, 2, 2, 1, 1, 0, 0, 2], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_ab0522bd77711af7608bf58a38ea5c7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe23d8d061a6333391c64fdc7b2deeb7
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 2, 2, 1, 0, 2, 1, 0, 2, 2, 1, 0, 1, 1, 1, 2], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_f73a5508067f429de5ac39adaeec029e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe23d8d061a6333391c64fdc7b2deeb7
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 2, 2, 0, 1, 2, 0, 2, 1, 0, 1, 0, 2, 0, 0, 0], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d9aa525b4530c5a65272ea3f4c0e7123(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe23d8d061a6333391c64fdc7b2deeb7
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 1, 0, 0, 0, 0, 0, 2, 1, 2, 2, 0, 0, 0, 1, 1], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_870e6f5ea4863e20223dc6c045304719(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe23d8d061a6333391c64fdc7b2deeb7
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 0, 0, 2, 1, 1, 0, 2, 2, 2, 2, 2, 0, 0, 0, 2], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_da4800c6c2a2ac612136629ded723717(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_866d0b647e17768cf8900ace3c007516
    def get_inputs(self):
        return [
            paddle.uniform([185658], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_acce70503ec43908e593f848099255b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6df22063311e4421bbd3d492da35aca0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[185658], dtype='int32'),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_fb690f1ba023a8e2036cc2e03fe3581b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df5062ca967442de395dfb80f2285f9b
    def get_inputs(self):
        return [
            paddle.uniform([185658, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[9], [1], [6], [9], [2], [8], [2]], dtype='int64').reshape([7, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_fb690f1ba023a8e2036cc2e03fe3581b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df5062ca967442de395dfb80f2285f9b
    def get_inputs(self):
        return [
            paddle.uniform([185658, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[9], [1], [6], [9], [2], [8], [2]], dtype='int64').reshape([7, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_f2c5160af211b2ed20424181b8b20f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b486fb2cfac6a4b9624be2eab74dce91
    def get_inputs(self):
        return [
            paddle.uniform([7, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6]], dtype='int32').reshape([7, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_507fcbd6bbc3e03f6bab560b59e9ab97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f3c4fc99580f866c5e3798ec6571465
    def get_inputs(self):
        return [
            paddle.uniform([6, 256, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [1], [2], [3], [4], [5]], dtype='int32').reshape([6, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()