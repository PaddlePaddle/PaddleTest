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



class PrimitiveOp_539dd483bdb66f93ec0733b68c1034a7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2b3805972505f88b5e03c4d2f953ee3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_539dd483bdb66f93ec0733b68c1034a7
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([5, 3], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_2b3805972505f88b5e03c4d2f953ee3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_539dd483bdb66f93ec0733b68c1034a7
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([5, 3], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_0b276116b8e53574370d174a27dc93d3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dcd51e18e2981dc2f27eaf484603d822(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b276116b8e53574370d174a27dc93d3
    def get_inputs(self):
        return [
            paddle.uniform([100, 256, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[100, 1], dtype='int32'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_228686245c205bc9933fdd0e091f009b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_58f2ae8c584fa47d2db8cfa417e62d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_137f77bd8233d65afce2d0b0ec21c889(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_539dd483bdb66f93ec0733b68c1034a7
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 9], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_137f77bd8233d65afce2d0b0ec21c889(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_539dd483bdb66f93ec0733b68c1034a7
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2, 9], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c2f64a3125289d171d4f1d7c0f04dca1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b276116b8e53574370d174a27dc93d3
    def get_inputs(self):
        return [
            paddle.uniform([300, 256, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[300, 1], dtype='int32'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c2f64a3125289d171d4f1d7c0f04dca1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b276116b8e53574370d174a27dc93d3
    def get_inputs(self):
        return [
            paddle.uniform([300, 256, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[300, 1], dtype='int32'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_b84aa604bf9de54c604002fdb53f25a0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_59b23ff91567885bb6e63c37e75cafe7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b84aa604bf9de54c604002fdb53f25a0
    def get_inputs(self):
        return [
            paddle.uniform([217413], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_57fea4b6a98b21e61c5c8c47efa08b18(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_300ac1a2505ee72d6a1eef85e7ef7c72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57fea4b6a98b21e61c5c8c47efa08b18
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[217413], dtype='int32'),
            paddle.randint(low=0, high=2, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_5a420bb9fb552bebcb43128ed9cb7ae6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3af148728d582a7c39e7598ffa00e75b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a420bb9fb552bebcb43128ed9cb7ae6
    def get_inputs(self):
        return [
            paddle.uniform([217413, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[103, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_3af148728d582a7c39e7598ffa00e75b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a420bb9fb552bebcb43128ed9cb7ae6
    def get_inputs(self):
        return [
            paddle.uniform([217413, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[103, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c5c7534852000aef0d7b19b14a80d005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_4771d81d145761ca2be48b6393f4422a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_7f1e2e63b11ec5d2d6b936d21a8112ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_fb7290c9a3165d6369957f17b27171e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_314647822990d2c6c7125a0962de6ce0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_f4acd471e9dda8a7d933c2604ea74334(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_57f0a37b81e082c61d51da5b3f2357b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_346aeaa335c4bc02077beb1ef3af830e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_cf6039d37726da307e730a14bb2bb4d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_80b071625620202e40fdd495584ed7c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_9430bfa0e28f81f73775843cd6da7ff0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_e4b908f21376ffdcee9f4f7588bf3df0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_bbb85c48ec527a7b70b912c280fa0b65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_e3d594c27db33946a544c663ff3b8b4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_80dea2088a7c711a5f240832723330a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_3fefbc694cea876835370a9e31d741c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_7a1045afaa0a7aa77dce0e9337b78def(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_61388268f48d70a26b0432871ac5090f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_53f037b14885f2297d12059c446d4ee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_53f037b14885f2297d12059c446d4ee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_53f037b14885f2297d12059c446d4ee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_53f037b14885f2297d12059c446d4ee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_53f037b14885f2297d12059c446d4ee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_53f037b14885f2297d12059c446d4ee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_53f037b14885f2297d12059c446d4ee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_53f037b14885f2297d12059c446d4ee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_53f037b14885f2297d12059c446d4ee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_53f037b14885f2297d12059c446d4ee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_53f037b14885f2297d12059c446d4ee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_53f037b14885f2297d12059c446d4ee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_53f037b14885f2297d12059c446d4ee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_53f037b14885f2297d12059c446d4ee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_53f037b14885f2297d12059c446d4ee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_53f037b14885f2297d12059c446d4ee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228686245c205bc9933fdd0e091f009b
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_9c515183356cf0e1d11f29cdc83853fe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_85fb19356c8e3fab0d2cfc8385d0ba59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c515183356cf0e1d11f29cdc83853fe
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 5], dtype='int32').reshape([2]),
            paddle.randint(low=0, high=2, shape=[1002], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_78e7a6ec34facf1c5e7ac22872ade1cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b84aa604bf9de54c604002fdb53f25a0
    def get_inputs(self):
        return [
            paddle.uniform([123783], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_15a8cc16ae243dcaf105dc3096141bb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57fea4b6a98b21e61c5c8c47efa08b18
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[123783], dtype='int32'),
            paddle.randint(low=0, high=2, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c710b5bd40391ad0d012e40089c51649(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a420bb9fb552bebcb43128ed9cb7ae6
    def get_inputs(self):
        return [
            paddle.uniform([123783, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[84, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c710b5bd40391ad0d012e40089c51649(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a420bb9fb552bebcb43128ed9cb7ae6
    def get_inputs(self):
        return [
            paddle.uniform([123783, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[84, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dcd51e18e2981dc2f27eaf484603d822(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b276116b8e53574370d174a27dc93d3
    def get_inputs(self):
        return [
            paddle.uniform([100, 256, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[100, 1], dtype='int32'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_52ae35c10ae64c589b14716fc12fb373(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c515183356cf0e1d11f29cdc83853fe
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 2, 3, 5, 2, 4, 4, 1, 0, 6, 8, 6, 0, 6, 9, 3, 4, 9, 4, 0, 0, 7, 8, 6, 1, 9, 3], dtype='int32').reshape([27]),
            paddle.randint(low=0, high=2, shape=[1027], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_4f82940ac3e89fcff0c774a912ad61e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c515183356cf0e1d11f29cdc83853fe
    def get_inputs(self):
        return [
            paddle.to_tensor([9, 5], dtype='int32').reshape([2]),
            paddle.randint(low=0, high=2, shape=[2002], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_40d28d0da3125536bcd77b831cd2523e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c515183356cf0e1d11f29cdc83853fe
    def get_inputs(self):
        return [
            paddle.to_tensor([6, 0, 2, 8, 9, 6, 2, 5, 4, 0, 2, 4, 2, 2, 3, 5, 2, 4, 4, 1, 0], dtype='int32').reshape([21]),
            paddle.randint(low=0, high=2, shape=[1021], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()