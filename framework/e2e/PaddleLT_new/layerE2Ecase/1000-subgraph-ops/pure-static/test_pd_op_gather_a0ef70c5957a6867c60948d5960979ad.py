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



class PrimitiveOp_3ddebde075d0ebe56fdc5ec2fc779d72(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3ee82f78f2d65185ae0a25cc36bd0e31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ddebde075d0ebe56fdc5ec2fc779d72
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([5, 3], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_3ee82f78f2d65185ae0a25cc36bd0e31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ddebde075d0ebe56fdc5ec2fc779d72
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([5, 3], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_f839561a6f8b25fd3f0e332bd5d6f96f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 256, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[100, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e97fb8adb254a1e54dc8ac5d3c79128a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f839561a6f8b25fd3f0e332bd5d6f96f
    def get_inputs(self):
        return [
            paddle.uniform([100, 256, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[100, 1], dtype='int32'),
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


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8d99111e970981f1dd46a8b9b34f59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_2bdaa0438952e3bb4b39df463f0246be(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_70f0a8afa4aa6c406666dc2b8bebe832(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bdaa0438952e3bb4b39df463f0246be
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 9], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_70f0a8afa4aa6c406666dc2b8bebe832(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bdaa0438952e3bb4b39df463f0246be
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 9], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_cf817579bb8a2762768aa16116d24195(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300, 256, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[300, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_54724172ebf6bb411fd2d539b056154a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf817579bb8a2762768aa16116d24195
    def get_inputs(self):
        return [
            paddle.uniform([300, 256, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[300, 1], dtype='int32'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_54724172ebf6bb411fd2d539b056154a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf817579bb8a2762768aa16116d24195
    def get_inputs(self):
        return [
            paddle.uniform([300, 256, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[300, 1], dtype='int32'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_3bdeb7ac00a4edbd59bf69e63655e77f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[217413], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7d093db3392815510921642a6ff60100(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bdeb7ac00a4edbd59bf69e63655e77f
    def get_inputs(self):
        return [
            paddle.uniform([217413], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_36019249ee2d8875f89f7556768c1580(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[217413], dtype='int32'),
            paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e7b0567c8419c4e0665d3cdd6e522363(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36019249ee2d8875f89f7556768c1580
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[217413], dtype='int32'),
            paddle.randint(low=0, high=2, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_c205402b823905760facb533d36a46c5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[217413, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[103, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_17f2bd3d4a0d0a5de4197befff04fea4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c205402b823905760facb533d36a46c5
    def get_inputs(self):
        return [
            paddle.uniform([217413, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[103, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_17f2bd3d4a0d0a5de4197befff04fea4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c205402b823905760facb533d36a46c5
    def get_inputs(self):
        return [
            paddle.uniform([217413, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[103, 1], dtype='int64'),
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


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_36ba0a6e1442b16c841350d971f23ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_46b1209e352bef23143834fbaef57181(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[16], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0bfffb8e86add697a4a183e6893583a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b1209e352bef23143834fbaef57181
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_538daff7fe947c4c217560113d5327cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b1209e352bef23143834fbaef57181
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_190d12cc610c838fa7370e8825afce63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b1209e352bef23143834fbaef57181
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8fe253e4ba9168d9d91ebfc50ed32899(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b1209e352bef23143834fbaef57181
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_5aef30600b2ce154bebf5576f7f2f25c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b1209e352bef23143834fbaef57181
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_922651aed4c847932bc00fb0aa648bfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b1209e352bef23143834fbaef57181
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_eab461f5ae226955df1ee00cea827cc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b1209e352bef23143834fbaef57181
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c9350f1b7054e23a28e8b17c3429598c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b1209e352bef23143834fbaef57181
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_47ad99209d3f4f93b0cb0becbc2316ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b1209e352bef23143834fbaef57181
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_74f86572be7b90337760479357415fd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b1209e352bef23143834fbaef57181
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_2eda4f3bee61d3f842ce372a6ab6f282(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b1209e352bef23143834fbaef57181
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_84239786cf1117eed2f2d0007d659eb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b1209e352bef23143834fbaef57181
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_ebb31eae5cc892b6901fcad8b8e37f67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b1209e352bef23143834fbaef57181
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_46449958cacf6af8a8ca329ce67cb0aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b1209e352bef23143834fbaef57181
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_2ad7c78c062e883ef21bed008227f694(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b1209e352bef23143834fbaef57181
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c63f52016785202fa7e8f4550aaeb7a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b1209e352bef23143834fbaef57181
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1], dtype='int64').reshape([16]),
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


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d22c4deef59a540e5229b2423fc96045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_f023d7187a00b0f25fe22fa10547efd6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_96d9134e91d03c413e34503f72aa9889(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f023d7187a00b0f25fe22fa10547efd6
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_96d9134e91d03c413e34503f72aa9889(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f023d7187a00b0f25fe22fa10547efd6
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_96d9134e91d03c413e34503f72aa9889(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f023d7187a00b0f25fe22fa10547efd6
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_96d9134e91d03c413e34503f72aa9889(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f023d7187a00b0f25fe22fa10547efd6
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_96d9134e91d03c413e34503f72aa9889(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f023d7187a00b0f25fe22fa10547efd6
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_96d9134e91d03c413e34503f72aa9889(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f023d7187a00b0f25fe22fa10547efd6
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_96d9134e91d03c413e34503f72aa9889(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f023d7187a00b0f25fe22fa10547efd6
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_96d9134e91d03c413e34503f72aa9889(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f023d7187a00b0f25fe22fa10547efd6
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_96d9134e91d03c413e34503f72aa9889(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f023d7187a00b0f25fe22fa10547efd6
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_96d9134e91d03c413e34503f72aa9889(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f023d7187a00b0f25fe22fa10547efd6
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_96d9134e91d03c413e34503f72aa9889(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f023d7187a00b0f25fe22fa10547efd6
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_96d9134e91d03c413e34503f72aa9889(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f023d7187a00b0f25fe22fa10547efd6
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_96d9134e91d03c413e34503f72aa9889(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f023d7187a00b0f25fe22fa10547efd6
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_96d9134e91d03c413e34503f72aa9889(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f023d7187a00b0f25fe22fa10547efd6
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_96d9134e91d03c413e34503f72aa9889(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f023d7187a00b0f25fe22fa10547efd6
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_96d9134e91d03c413e34503f72aa9889(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f023d7187a00b0f25fe22fa10547efd6
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_4cb27e8bc5c6185d340d34347652e6db(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2], dtype='int32'),
            paddle.static.InputSpec(shape=[1002], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_92bea91585756f803f32b339557698d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4cb27e8bc5c6185d340d34347652e6db
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 5], dtype='int32').reshape([2]),
            paddle.randint(low=0, high=2, shape=[1002], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_cd917d05b14ba00339d5da1e32c02208(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[123783], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_46c52b47fa3a8a4829292aef62c80fba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd917d05b14ba00339d5da1e32c02208
    def get_inputs(self):
        return [
            paddle.uniform([123783], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_71b1c26c5e83573a5f2db497c5c23fb4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[123783], dtype='int32'),
            paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_646808b8ee62f694476232b888892864(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71b1c26c5e83573a5f2db497c5c23fb4
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[123783], dtype='int32'),
            paddle.randint(low=0, high=2, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_fcc3afcf0c994a5eb398628dd86e32be(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[123783, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[84, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_09c96b9a5b933640cbc1c5cd8ecfb504(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fcc3afcf0c994a5eb398628dd86e32be
    def get_inputs(self):
        return [
            paddle.uniform([123783, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[84, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_09c96b9a5b933640cbc1c5cd8ecfb504(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fcc3afcf0c994a5eb398628dd86e32be
    def get_inputs(self):
        return [
            paddle.uniform([123783, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[84, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_e97fb8adb254a1e54dc8ac5d3c79128a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f839561a6f8b25fd3f0e332bd5d6f96f
    def get_inputs(self):
        return [
            paddle.uniform([100, 256, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[100, 1], dtype='int32'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_80b4b2cb8a58b86dc2f73cb50e7516c7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[27], dtype='int32'),
            paddle.static.InputSpec(shape=[1027], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_009aac4403edc9de502333943c785f08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80b4b2cb8a58b86dc2f73cb50e7516c7
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 2, 3, 5, 2, 4, 4, 1, 0, 6, 8, 6, 0, 6, 9, 3, 4, 9, 4, 0, 0, 7, 8, 6, 1, 9, 3], dtype='int32').reshape([27]),
            paddle.randint(low=0, high=2, shape=[1027], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_d4d14f2da2f8f848be987540fdf91294(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2], dtype='int32'),
            paddle.static.InputSpec(shape=[2002], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_81b72f30f2228e2c08995bbf853c609e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d14f2da2f8f848be987540fdf91294
    def get_inputs(self):
        return [
            paddle.to_tensor([9, 5], dtype='int32').reshape([2]),
            paddle.randint(low=0, high=2, shape=[2002], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_133238ab6cd6751fd5248f320291db40(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[21], dtype='int32'),
            paddle.static.InputSpec(shape=[1021], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_490bc1e126775ae61515be6c503d347c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_133238ab6cd6751fd5248f320291db40
    def get_inputs(self):
        return [
            paddle.to_tensor([6, 0, 2, 8, 9, 6, 2, 5, 4, 0, 2, 4, 2, 2, 3, 5, 2, 4, 4, 1, 0], dtype='int32').reshape([21]),
            paddle.randint(low=0, high=2, shape=[1021], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()