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



class PrimitiveOp_228fdc912558e98bd21a04f8a2a84977(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 576, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f94e0924fcf405cac7f78c21754fd5d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228fdc912558e98bd21a04f8a2a84977
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_9631aec128c9d55dc0e417cdd91613fa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 48, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7f5fb00860ce1e36fe2575b4ab8414fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9631aec128c9d55dc0e417cdd91613fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_ea1b9147a2ba0a1f0e44a663741d784a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_30d9a2273f71450ff2f48b7dcf1a0498(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea1b9147a2ba0a1f0e44a663741d784a
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_8612d0d68b8a650001a86691bd316818(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 23, 23, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 23, 23, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8c07ea6383662793c01e8d4a939dac4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8612d0d68b8a650001a86691bd316818
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8c07ea6383662793c01e8d4a939dac4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8612d0d68b8a650001a86691bd316818
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8c07ea6383662793c01e8d4a939dac4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8612d0d68b8a650001a86691bd316818
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8c07ea6383662793c01e8d4a939dac4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8612d0d68b8a650001a86691bd316818
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8c07ea6383662793c01e8d4a939dac4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8612d0d68b8a650001a86691bd316818
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8c07ea6383662793c01e8d4a939dac4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8612d0d68b8a650001a86691bd316818
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_569c7b46c0422c9cc58e3eb62805439e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 960, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_549018118c774c5ea626225dd22333f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_569c7b46c0422c9cc58e3eb62805439e
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_be5ae9c4245c383ff6a73071a8803822(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 672, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cf887f402a2f49136bc1ced9444f50e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be5ae9c4245c383ff6a73071a8803822
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7f5fb00860ce1e36fe2575b4ab8414fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9631aec128c9d55dc0e417cdd91613fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_30d9a2273f71450ff2f48b7dcf1a0498(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea1b9147a2ba0a1f0e44a663741d784a
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_8e31c2ef7375f16acdd38efc5dec8d32(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4a3628e98beb097e056490e649aef4e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e31c2ef7375f16acdd38efc5dec8d32
    def get_inputs(self):
        return [
            paddle.uniform([22, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 60], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_fdfc90c9e2c9a224a34d7c28b1a71a8e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 400, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0342f6fdbef59c291fde2cbee699f1b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fdfc90c9e2c9a224a34d7c28b1a71a8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4a4ed60cdbbe5bf6cefc6d11b72b6475(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4a4ed60cdbbe5bf6cefc6d11b72b6475(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_692eb51a1b03665b520f0b424ab1b579(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9f305d81060aa48303b1d666a49e40a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_692eb51a1b03665b520f0b424ab1b579
    def get_inputs(self):
        return [
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9f305d81060aa48303b1d666a49e40a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_692eb51a1b03665b520f0b424ab1b579
    def get_inputs(self):
        return [
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4a4ed60cdbbe5bf6cefc6d11b72b6475(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4a4ed60cdbbe5bf6cefc6d11b72b6475(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b37e5dfec6ab77274ff6dfd673c58072(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3024, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3024, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ff3f4b93776b18d160386f5d0d27c1d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b37e5dfec6ab77274ff6dfd673c58072
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3024, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1934b19eec6f2d63afc5b3cb82107f16(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 672], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 672], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0318b881e6a6ae476f295ae0cadf46da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1934b19eec6f2d63afc5b3cb82107f16
    def get_inputs(self):
        return [
            paddle.uniform([1, 672], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 672], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_880399ea48e19e5f1300b839c37f749b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_880399ea48e19e5f1300b839c37f749b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e4d5a020168a344d30c52f20fd38723f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_692eb51a1b03665b520f0b424ab1b579
    def get_inputs(self):
        return [
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e4d5a020168a344d30c52f20fd38723f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_692eb51a1b03665b520f0b424ab1b579
    def get_inputs(self):
        return [
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_880399ea48e19e5f1300b839c37f749b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_880399ea48e19e5f1300b839c37f749b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_78a3f5423a12db24344665283d140ff5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4725, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4725, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_836e0f8fbd14968820eda8993a98f903(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78a3f5423a12db24344665283d140ff5
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4725, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cf887f402a2f49136bc1ced9444f50e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be5ae9c4245c383ff6a73071a8803822
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_61a593482a75fcd987aa4aed1be4abf9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 768, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_88f101d3b83ff8bf4eb6cd4531f0e5f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61a593482a75fcd987aa4aed1be4abf9
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_c58b10dd5ec878aedaeec3d45c165a9b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 480], dtype='float32'),
            paddle.static.InputSpec(shape=[171, 480], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_616f6bcc4dd3ee9ec202f88917e8cb7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c58b10dd5ec878aedaeec3d45c165a9b
    def get_inputs(self):
        return [
            paddle.uniform([171, 480], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([171, 480], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_30d9a2273f71450ff2f48b7dcf1a0498(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea1b9147a2ba0a1f0e44a663741d784a
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_852b63e07076200b2dd5a3d370d7746e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 72, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_65a393bdcb772f4c946958199883a04a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_852b63e07076200b2dd5a3d370d7746e
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_ce23a53a29ef92e6eec8d8a549a7d85f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6d94d45888e5de3e09ab9696b2370b39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce23a53a29ef92e6eec8d8a549a7d85f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_5ef48365d12968559582b82ede7e159a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 12, 12, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 12, 12, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6efa63011d490480fd71159b54a6142a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ef48365d12968559582b82ede7e159a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6efa63011d490480fd71159b54a6142a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ef48365d12968559582b82ede7e159a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6efa63011d490480fd71159b54a6142a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ef48365d12968559582b82ede7e159a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6efa63011d490480fd71159b54a6142a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ef48365d12968559582b82ede7e159a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6efa63011d490480fd71159b54a6142a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ef48365d12968559582b82ede7e159a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6efa63011d490480fd71159b54a6142a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ef48365d12968559582b82ede7e159a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e485e8990d1a5ae2479706aefca977fb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 240, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d8394585742f6b0e295abbb86b1425e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e485e8990d1a5ae2479706aefca977fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_5509f5aa0a656ae8da055837ded388b1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[145, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c6d3e7812f5b224c3325659206e45aad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5509f5aa0a656ae8da055837ded388b1
    def get_inputs(self):
        return [
            paddle.uniform([145, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([145, 60], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ea88ac253d18142d28130c6adb0c2cf4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6b0ed726f007dff76630976578469269(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea88ac253d18142d28130c6adb0c2cf4
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4195559024810791], [0.258736252784729], [0.4700632691383362], [-0.007235139608383179]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.2825867533683777], [-0.4766601324081421], [-0.12349918484687805], [-0.017541974782943726]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_b2593d33968332b38a016ec1498d8820(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea88ac253d18142d28130c6adb0c2cf4
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1602063775062561], [-0.14480668306350708], [0.40289902687072754], [-0.3244452476501465]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.38134509325027466], [0.24650567770004272], [0.34999436140060425], [-0.4619288444519043]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_cf0a1eb460a4653a63ac32db4322f570(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea88ac253d18142d28130c6adb0c2cf4
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3121303915977478], [-0.5012533068656921], [-0.10262379050254822], [0.3540444076061249]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_9a16620bbb790aee121b232d0e7877b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea88ac253d18142d28130c6adb0c2cf4
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1762411892414093], [-0.10018911957740784], [0.4803326427936554], [-0.9249767661094666]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_67ae6e1445efa103fb9c0b6d79cac0a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea88ac253d18142d28130c6adb0c2cf4
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.42702656984329224], [-0.03797486424446106], [-0.32467591762542725], [-0.3715863823890686]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.029543638229370117], [0.02459317445755005], [-0.020875394344329834], [-0.43152952194213867]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_4895573a3999fb5a0f6fc942e4046210(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea88ac253d18142d28130c6adb0c2cf4
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.016034811735153198], [-0.04461756348609924], [-0.13033828139305115], [0.1816677451133728]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.29278564453125], [-0.1465194821357727], [-0.4884662628173828], [0.46304792165756226]], dtype='float32').reshape([4, 1]),
        ]


class PrimitiveOp_f5feed18fa2e71b8e3dabb3b5beb0932(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[6, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_10bc453a7c1ed4a2eb8fa7974deedee5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5feed18fa2e71b8e3dabb3b5beb0932
    def get_inputs(self):
        return [
            paddle.to_tensor([[[-0.12320411205291748]], [[0.42164498567581177]], [[0.35338306427001953]], [[0.020320534706115723]], [[-0.03214597702026367]], [[-0.30578428506851196]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[4.135169982910156]], [[4.135169982910156]], [[4.135169982910156]], [[4.135169982910156]], [[4.135169982910156]], [[4.135169982910156]]], dtype='float32').reshape([6, 1, 1]),
        ]


class PrimitiveOp_77c8939003db30e283086fd87b133983(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_66be25e13d2819d0e703c5c4a5fffa3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8939003db30e283086fd87b133983
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_24b609e55c8ed8004a7da1a199591925(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c291ff260df6a6c5d2b3fde3b54bce2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24b609e55c8ed8004a7da1a199591925
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_65a393bdcb772f4c946958199883a04a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_852b63e07076200b2dd5a3d370d7746e
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_65a393bdcb772f4c946958199883a04a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_852b63e07076200b2dd5a3d370d7746e
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_3b6f51e0ff307ce9e9ae0a6d3c1cc52e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6], dtype='float32'),
            paddle.static.InputSpec(shape=[6], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4e71990ce650a1f8b0d7db56a16a98be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b6f51e0ff307ce9e9ae0a6d3c1cc52e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.32623517513275146, 0.2671630382537842, 0.27793484926223755, 0.19273710250854492, 0.4619840383529663, 0.17146217823028564], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.270524263381958, -0.20599150657653809, -0.06320270895957947, -0.2330784797668457, -0.28073909878730774, 0.2532276511192322], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_693eba8b593d3e54cb85128c5ad89447(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b6f51e0ff307ce9e9ae0a6d3c1cc52e
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.21393254399299622, 0.34922271966934204, 0.2899574637413025, 0.3019695281982422, 0.06090348958969116, 0.11269927024841309], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.19772139191627502, 0.4174601435661316, 0.029941082000732422, -0.15820688009262085, -0.4392582178115845, -0.4687579572200775], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_09a6e74ec9296d644222fc0c0a894e3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b6f51e0ff307ce9e9ae0a6d3c1cc52e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.32623517513275146, -0.24192336201667786, 0.27793484926223755, 0.19273710250854492, 0.4619840383529663, 0.057091474533081055], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.03316903114318848, 0.37095922231674194, 0.48807185888290405, 0.02101588249206543, -0.059602320194244385, 0.2165842056274414], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_f41ca1ba41ef55795f2c96785944ba30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b6f51e0ff307ce9e9ae0a6d3c1cc52e
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.48779433965682983, 0.34922271966934204, -0.4023532271385193, 0.3019695281982422, 0.06090348958969116, 0.11269927024841309], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.03374040126800537, -0.1420654058456421, -0.10500526428222656, -0.35081708431243896, -0.40645480155944824, -0.12897560000419617], dtype='float32').reshape([6]),
        ]


class PrimitiveOp_8052a12c240ca6fb659f402a78b33975(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 320, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3e9b9c62a146fa260bf0d800de00f030(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8052a12c240ca6fb659f402a78b33975
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_55abc8d01f55954b67368efa6cc544b8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 120, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e6693535d43d87ffd0285ad3e48035fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55abc8d01f55954b67368efa6cc544b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_f70737c980555b1f1ef7a501e8729c6e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 336, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dee5e4948691f7bf1973b4c1b54e8cc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f70737c980555b1f1ef7a501e8729c6e
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_45f5cf2c205f6ee7828e9c2334b2f164(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_652e9b65f6217f468c2017e65a56049e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45f5cf2c205f6ee7828e9c2334b2f164
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7f5fb00860ce1e36fe2575b4ab8414fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9631aec128c9d55dc0e417cdd91613fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e6693535d43d87ffd0285ad3e48035fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55abc8d01f55954b67368efa6cc544b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3e9b9c62a146fa260bf0d800de00f030(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8052a12c240ca6fb659f402a78b33975
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_cf887f402a2f49136bc1ced9444f50e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be5ae9c4245c383ff6a73071a8803822
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_de0b9afc4a25723de28c49a95ea6682e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 384, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_25bacded1ecad1d067e3f97121f1bcfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de0b9afc4a25723de28c49a95ea6682e
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0342f6fdbef59c291fde2cbee699f1b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fdfc90c9e2c9a224a34d7c28b1a71a8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_65a393bdcb772f4c946958199883a04a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_852b63e07076200b2dd5a3d370d7746e
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_0feea93e0825a4843b257e893028a9a9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 336], dtype='float32'),
            paddle.static.InputSpec(shape=[145, 336], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a23af7c7fee607c7c9349718dc14a5c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0feea93e0825a4843b257e893028a9a9
    def get_inputs(self):
        return [
            paddle.uniform([145, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([145, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_92a93e26fdabbeac29ed6b50f4b9a312(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 46, 46, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 46, 46, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fc4fbe401107c1bc8a817becc87e36f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92a93e26fdabbeac29ed6b50f4b9a312
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc4fbe401107c1bc8a817becc87e36f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92a93e26fdabbeac29ed6b50f4b9a312
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc4fbe401107c1bc8a817becc87e36f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92a93e26fdabbeac29ed6b50f4b9a312
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc4fbe401107c1bc8a817becc87e36f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92a93e26fdabbeac29ed6b50f4b9a312
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc4fbe401107c1bc8a817becc87e36f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92a93e26fdabbeac29ed6b50f4b9a312
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc4fbe401107c1bc8a817becc87e36f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92a93e26fdabbeac29ed6b50f4b9a312
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0b74dfbe80ca4eee035b1a8a60be23b9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 76, 76, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 76, 76, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0fb232b9dc0b7db8c6b2cf623180552b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b74dfbe80ca4eee035b1a8a60be23b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0fb232b9dc0b7db8c6b2cf623180552b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b74dfbe80ca4eee035b1a8a60be23b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0fb232b9dc0b7db8c6b2cf623180552b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b74dfbe80ca4eee035b1a8a60be23b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0fb232b9dc0b7db8c6b2cf623180552b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b74dfbe80ca4eee035b1a8a60be23b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0fb232b9dc0b7db8c6b2cf623180552b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b74dfbe80ca4eee035b1a8a60be23b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0fb232b9dc0b7db8c6b2cf623180552b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b74dfbe80ca4eee035b1a8a60be23b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f94e0924fcf405cac7f78c21754fd5d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228fdc912558e98bd21a04f8a2a84977
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_841af32782ad17131f2ec7454f5a43fa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 240], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 240], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0e1bec2c950e6f520bdf27cff931f206(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_841af32782ad17131f2ec7454f5a43fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 240], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d8394585742f6b0e295abbb86b1425e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e485e8990d1a5ae2479706aefca977fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c291ff260df6a6c5d2b3fde3b54bce2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24b609e55c8ed8004a7da1a199591925
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_65a393bdcb772f4c946958199883a04a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_852b63e07076200b2dd5a3d370d7746e
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_30d9a2273f71450ff2f48b7dcf1a0498(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea1b9147a2ba0a1f0e44a663741d784a
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_b812e843f25f9f629b3ed8ee38ba590d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 92, 92, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 92, 92, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cd502a4ab22691e7a36da5067b3f8ed7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b812e843f25f9f629b3ed8ee38ba590d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cd502a4ab22691e7a36da5067b3f8ed7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b812e843f25f9f629b3ed8ee38ba590d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cd502a4ab22691e7a36da5067b3f8ed7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b812e843f25f9f629b3ed8ee38ba590d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cd502a4ab22691e7a36da5067b3f8ed7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b812e843f25f9f629b3ed8ee38ba590d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cd502a4ab22691e7a36da5067b3f8ed7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b812e843f25f9f629b3ed8ee38ba590d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cd502a4ab22691e7a36da5067b3f8ed7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b812e843f25f9f629b3ed8ee38ba590d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6d94d45888e5de3e09ab9696b2370b39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce23a53a29ef92e6eec8d8a549a7d85f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_05baf4743ceb3c4812ec461ba7942f94(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[6, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_32736e117af4056e68a310161638c4c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05baf4743ceb3c4812ec461ba7942f94
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2814381718635559], [0.3015877604484558], [-0.2885453701019287], [-0.31574276089668274], [0.2026931643486023], [-0.0327763557434082]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.10710150003433228], [0.4202950596809387], [0.20075774192810059], [-0.24555674195289612], [-0.39915066957473755], [-0.2773142158985138]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_dc902c29b195a0dae65c428f97fbdafa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05baf4743ceb3c4812ec461ba7942f94
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.45480841398239136], [-0.3500272035598755], [-0.0469474196434021], [0.07794475555419922], [0.19924408197402954], [-0.2040221095085144]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.4103016257286072], [-0.2566860318183899], [0.030750691890716553], [-0.09059321880340576], [-0.05198678374290466], [0.11784225702285767]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_c88e352b732c3c994f9d867bcab82335(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05baf4743ceb3c4812ec461ba7942f94
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.2527996897697449], [-0.1516473889350891], [-0.6138649582862854], [-0.5521718263626099], [-0.6273437142372131], [-0.37911269068717957]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_4e8b2b7ee137d2d308960c04ccef25ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05baf4743ceb3c4812ec461ba7942f94
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.04148709774017334], [-0.7964882254600525], [-0.29898810386657715], [-0.2718324661254883], [-0.0077477991580963135], [-0.23272746801376343]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_69e4362380ff032772a2a3cacd2caebf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05baf4743ceb3c4812ec461ba7942f94
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.35990118980407715], [0.4532351493835449], [0.3253195881843567], [-0.18913787603378296], [0.22819304466247559], [0.10179847478866577]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.34053710103034973], [0.2731626033782959], [-0.39370104670524597], [0.23642903566360474], [-0.17116469144821167], [-0.068158358335495]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_531213c2b233bd5a446152c3fa1be891(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05baf4743ceb3c4812ec461ba7942f94
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.36881452798843384], [0.446461021900177], [0.25204068422317505], [0.18123924732208252], [-0.04423898458480835], [0.028705358505249023]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.4686683714389801], [0.3912338614463806], [-0.45743855834007263], [-0.008816301822662354], [-0.15691471099853516], [-0.17567184567451477]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_549018118c774c5ea626225dd22333f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_569c7b46c0422c9cc58e3eb62805439e
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c291ff260df6a6c5d2b3fde3b54bce2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24b609e55c8ed8004a7da1a199591925
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d8394585742f6b0e295abbb86b1425e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e485e8990d1a5ae2479706aefca977fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_65a393bdcb772f4c946958199883a04a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_852b63e07076200b2dd5a3d370d7746e
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_9ac220c649dd3c8ac3e434814385a8e3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 480, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2f99997c92443c919b5477a750521337(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac220c649dd3c8ac3e434814385a8e3
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_cf887f402a2f49136bc1ced9444f50e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be5ae9c4245c383ff6a73071a8803822
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_0d78d8de419b03854af5c7f9bc2f0422(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 44, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_18d78562d215460cc83c25380db10a2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d78d8de419b03854af5c7f9bc2f0422
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c291ff260df6a6c5d2b3fde3b54bce2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24b609e55c8ed8004a7da1a199591925
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6d94d45888e5de3e09ab9696b2370b39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce23a53a29ef92e6eec8d8a549a7d85f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_ce18fb8c6d189e028ab68300eb069935(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 56, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_09c85685f7f3b4e5a77689eea932af4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce18fb8c6d189e028ab68300eb069935
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_549018118c774c5ea626225dd22333f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_569c7b46c0422c9cc58e3eb62805439e
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_f1ee98d99e145aefac757d4fb583eb4a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 336], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 336], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a461680d38504b648a1f24890fb9d0b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1ee98d99e145aefac757d4fb583eb4a
    def get_inputs(self):
        return [
            paddle.uniform([22, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_25bacded1ecad1d067e3f97121f1bcfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de0b9afc4a25723de28c49a95ea6682e
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6d94d45888e5de3e09ab9696b2370b39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce23a53a29ef92e6eec8d8a549a7d85f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_eae91c912a1875b674a78db7021db3d8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4d02334004a436706c1573638fc5c351(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eae91c912a1875b674a78db7021db3d8
    def get_inputs(self):
        return [
            paddle.uniform([10, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 60], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_efb1e03bb22a924dac4f6d857df404f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_27026c3313adeea4348d27a5034b2c14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb1e03bb22a924dac4f6d857df404f9
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.6145999431610107]], [[0.6093606948852539]], [[0.4309464395046234]], [[0.5707991719245911]], [[0.5129478573799133]], [[0.5181457996368408]], [[0.5002401471138]], [[0.5722014904022217]], [[0.5123381614685059]], [[0.410256028175354]], [[0.4763246178627014]], [[0.4863011837005615]], [[0.5800120830535889]], [[0.3538127541542053]], [[0.5231696963310242]], [[0.5127725601196289]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_549018118c774c5ea626225dd22333f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_569c7b46c0422c9cc58e3eb62805439e
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_25bacded1ecad1d067e3f97121f1bcfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de0b9afc4a25723de28c49a95ea6682e
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_65a393bdcb772f4c946958199883a04a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_852b63e07076200b2dd5a3d370d7746e
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2f99997c92443c919b5477a750521337(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac220c649dd3c8ac3e434814385a8e3
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_5486865b270c0886eb6434b61235e938(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_824452bb148158ce59a7138c4286fd30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5486865b270c0886eb6434b61235e938
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d8394585742f6b0e295abbb86b1425e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e485e8990d1a5ae2479706aefca977fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_65a393bdcb772f4c946958199883a04a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_852b63e07076200b2dd5a3d370d7746e
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c291ff260df6a6c5d2b3fde3b54bce2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24b609e55c8ed8004a7da1a199591925
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_3213aced3839e6653017d5749c904e32(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 24, 24, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 24, 24, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_338f4762f367f6e85f71b245f764c715(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3213aced3839e6653017d5749c904e32
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_338f4762f367f6e85f71b245f764c715(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3213aced3839e6653017d5749c904e32
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_338f4762f367f6e85f71b245f764c715(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3213aced3839e6653017d5749c904e32
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_338f4762f367f6e85f71b245f764c715(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3213aced3839e6653017d5749c904e32
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_338f4762f367f6e85f71b245f764c715(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3213aced3839e6653017d5749c904e32
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_338f4762f367f6e85f71b245f764c715(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3213aced3839e6653017d5749c904e32
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_25bacded1ecad1d067e3f97121f1bcfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de0b9afc4a25723de28c49a95ea6682e
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c291ff260df6a6c5d2b3fde3b54bce2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24b609e55c8ed8004a7da1a199591925
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_88f101d3b83ff8bf4eb6cd4531f0e5f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61a593482a75fcd987aa4aed1be4abf9
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_4a341aca91d9c4fe08f1de21b5686abc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 960], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 960], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c787589fa4470c6e4d9e8b5d86598910(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a341aca91d9c4fe08f1de21b5686abc
    def get_inputs(self):
        return [
            paddle.uniform([1, 960], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 960], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a860fac71210374cca5ca09f1371af03(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 19, 19, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 19, 19, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4c700927289cf3a5120debd41da1e8c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a860fac71210374cca5ca09f1371af03
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4c700927289cf3a5120debd41da1e8c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a860fac71210374cca5ca09f1371af03
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4c700927289cf3a5120debd41da1e8c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a860fac71210374cca5ca09f1371af03
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4c700927289cf3a5120debd41da1e8c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a860fac71210374cca5ca09f1371af03
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4c700927289cf3a5120debd41da1e8c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a860fac71210374cca5ca09f1371af03
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4c700927289cf3a5120debd41da1e8c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a860fac71210374cca5ca09f1371af03
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dee5e4948691f7bf1973b4c1b54e8cc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f70737c980555b1f1ef7a501e8729c6e
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_dd66883f5c0b8e0af0d86ff7fd9e72e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dd66883f5c0b8e0af0d86ff7fd9e72e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cc1e6c8bdb642be288ea7240ccbb8509(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_692eb51a1b03665b520f0b424ab1b579
    def get_inputs(self):
        return [
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cc1e6c8bdb642be288ea7240ccbb8509(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_692eb51a1b03665b520f0b424ab1b579
    def get_inputs(self):
        return [
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dd66883f5c0b8e0af0d86ff7fd9e72e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dd66883f5c0b8e0af0d86ff7fd9e72e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_8837995068d85ed2037c87dda63d6553(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4116, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4116, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b18b05307e30de4e0b0d8ff0ac8dc3e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8837995068d85ed2037c87dda63d6553
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4116, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b629aeb02983c44eb0a04a805e9a408d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 21, 21, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 21, 21, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_574a90fd88bca8631187ac8eac5483b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b629aeb02983c44eb0a04a805e9a408d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_574a90fd88bca8631187ac8eac5483b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b629aeb02983c44eb0a04a805e9a408d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_574a90fd88bca8631187ac8eac5483b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b629aeb02983c44eb0a04a805e9a408d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_574a90fd88bca8631187ac8eac5483b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b629aeb02983c44eb0a04a805e9a408d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_574a90fd88bca8631187ac8eac5483b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b629aeb02983c44eb0a04a805e9a408d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_574a90fd88bca8631187ac8eac5483b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b629aeb02983c44eb0a04a805e9a408d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1b8c32d5f8aba20d6b502faaf4ab36c3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 624], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 624], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9a9ac4cbe01d0c86723997fc83454f64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b8c32d5f8aba20d6b502faaf4ab36c3
    def get_inputs(self):
        return [
            paddle.uniform([1, 624], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 624], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cf887f402a2f49136bc1ced9444f50e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be5ae9c4245c383ff6a73071a8803822
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_8536edfb5dd3539b3c23bcac2ba62069(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 100, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a321a9c2926e9f13265fb8e9a7c2bb02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8536edfb5dd3539b3c23bcac2ba62069
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_88f101d3b83ff8bf4eb6cd4531f0e5f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61a593482a75fcd987aa4aed1be4abf9
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_423e2ebb1b58e32bb30759dd62f4168e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 200, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2b1576864577ae0be06e1f7b97aa4523(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_423e2ebb1b58e32bb30759dd62f4168e
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_a3e6391bedc81e3bd11aff367573ac5a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[171, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5d45d3e2f8547209e9aa30fb07edfea7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3e6391bedc81e3bd11aff367573ac5a
    def get_inputs(self):
        return [
            paddle.uniform([171, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([171, 60], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_824452bb148158ce59a7138c4286fd30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5486865b270c0886eb6434b61235e938
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_0f682df592bb7c12f6b95a44b6278df0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 336], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 336], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_03ec3ae6ed22643ce29fbc10add60f73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f682df592bb7c12f6b95a44b6278df0
    def get_inputs(self):
        return [
            paddle.uniform([10, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_cdba59fb7e86bd9a3f073cbfd82e0b47(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 42, 42, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 42, 42, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_307ec8e943a596446e9d5cf788cf8560(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdba59fb7e86bd9a3f073cbfd82e0b47
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_307ec8e943a596446e9d5cf788cf8560(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdba59fb7e86bd9a3f073cbfd82e0b47
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_307ec8e943a596446e9d5cf788cf8560(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdba59fb7e86bd9a3f073cbfd82e0b47
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_307ec8e943a596446e9d5cf788cf8560(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdba59fb7e86bd9a3f073cbfd82e0b47
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_307ec8e943a596446e9d5cf788cf8560(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdba59fb7e86bd9a3f073cbfd82e0b47
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_307ec8e943a596446e9d5cf788cf8560(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdba59fb7e86bd9a3f073cbfd82e0b47
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7f5fb00860ce1e36fe2575b4ab8414fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9631aec128c9d55dc0e417cdd91613fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_d02f3dd6e00738acfd7d838605f7bb5f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 480], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 480], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7efe2dc3724db45a6da8d9674ded7da6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d02f3dd6e00738acfd7d838605f7bb5f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 480], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c291ff260df6a6c5d2b3fde3b54bce2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24b609e55c8ed8004a7da1a199591925
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_9c2bda744f644ff450a8fa8efb65df70(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[145, 36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f41d02074d6f4466a8d3c7473f8f8545(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c2bda744f644ff450a8fa8efb65df70
    def get_inputs(self):
        return [
            paddle.uniform([145, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([145, 36], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_2850aeb421d78f007bb2e5ce26a7c30c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 84, 84, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 84, 84, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f263286b03f2f30037f5f269fbeee3f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2850aeb421d78f007bb2e5ce26a7c30c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f263286b03f2f30037f5f269fbeee3f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2850aeb421d78f007bb2e5ce26a7c30c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f263286b03f2f30037f5f269fbeee3f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2850aeb421d78f007bb2e5ce26a7c30c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f263286b03f2f30037f5f269fbeee3f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2850aeb421d78f007bb2e5ce26a7c30c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f263286b03f2f30037f5f269fbeee3f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2850aeb421d78f007bb2e5ce26a7c30c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f263286b03f2f30037f5f269fbeee3f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2850aeb421d78f007bb2e5ce26a7c30c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_03ec3ae6ed22643ce29fbc10add60f73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f682df592bb7c12f6b95a44b6278df0
    def get_inputs(self):
        return [
            paddle.uniform([10, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c291ff260df6a6c5d2b3fde3b54bce2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24b609e55c8ed8004a7da1a199591925
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_c8e51305a7ff4a6086a9932522e3ba63(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[5, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ab96379d301cb1ece7fadc593e81d00c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8e51305a7ff4a6086a9932522e3ba63
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.2732882499694824], [0.21844327449798584], [0.1365358829498291], [0.1824830174446106], [0.38517266511917114]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.32311898469924927], [-0.4162471294403076], [-0.24600857496261597], [-0.4810544550418854], [0.46119225025177]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_37be751bab8e459f71fdcd81c683ffd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8e51305a7ff4a6086a9932522e3ba63
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.47203218936920166], [0.33248358964920044], [-0.021406471729278564], [-0.28133857250213623], [0.33753883838653564]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.1121317446231842], [0.17175018787384033], [0.2869950532913208], [-0.10179561376571655], [-0.06455051898956299]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_d3bf724fc3bcc92948a8f32e04ac1df5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8e51305a7ff4a6086a9932522e3ba63
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.6303538680076599], [-0.7695514559745789], [-0.234910786151886], [-0.6752417087554932], [0.5591869354248047]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_73f6c733dbc2cc6464e85ea93c09d045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8e51305a7ff4a6086a9932522e3ba63
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.13120320439338684], [0.1371973752975464], [-0.3553307056427002], [-0.1541731357574463], [-0.47886979579925537]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_65a8a0de59bb5d51db0ae7d609bc10d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8e51305a7ff4a6086a9932522e3ba63
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.11598128080368042], [-0.19975826144218445], [-0.01109778881072998], [0.1941872239112854], [-0.17401427030563354]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.30723488330841064], [0.35330432653427124], [-0.4845125675201416], [-0.1398959755897522], [-0.31891727447509766]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_0234acb14186dd68bd1a5964fb4c848c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8e51305a7ff4a6086a9932522e3ba63
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.49989163875579834], [-0.10271179676055908], [0.22155678272247314], [-0.12957623600959778], [0.27495449781417847]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.24333494901657104], [0.034552812576293945], [0.33392423391342163], [-0.12716543674468994], [0.4143192768096924]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_30d9a2273f71450ff2f48b7dcf1a0498(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea1b9147a2ba0a1f0e44a663741d784a
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_dd3ce5f558f138b64fec7fc0afd1e656(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dd3ce5f558f138b64fec7fc0afd1e656(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ff24a3997856f6cf242af6054e79f4ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_692eb51a1b03665b520f0b424ab1b579
    def get_inputs(self):
        return [
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ff24a3997856f6cf242af6054e79f4ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_692eb51a1b03665b520f0b424ab1b579
    def get_inputs(self):
        return [
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dd3ce5f558f138b64fec7fc0afd1e656(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dd3ce5f558f138b64fec7fc0afd1e656(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_4d934d40258676c7c08f228ee4092704(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3549, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f86d54552cdedd21652c12a9c5d77456(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d934d40258676c7c08f228ee4092704
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3549, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6d94d45888e5de3e09ab9696b2370b39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce23a53a29ef92e6eec8d8a549a7d85f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_cf887f402a2f49136bc1ced9444f50e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be5ae9c4245c383ff6a73071a8803822
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_b83a8d915b3bac88e594e9e88cc3be3c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 72], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 72], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8a3fedef1500210ad8b98dfa687ebe9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b83a8d915b3bac88e594e9e88cc3be3c
    def get_inputs(self):
        return [
            paddle.uniform([1, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 72], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_65a393bdcb772f4c946958199883a04a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_852b63e07076200b2dd5a3d370d7746e
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c291ff260df6a6c5d2b3fde3b54bce2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24b609e55c8ed8004a7da1a199591925
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_25bacded1ecad1d067e3f97121f1bcfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de0b9afc4a25723de28c49a95ea6682e
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_cb2f53adb77a2b3ae687dc6d6f3ab6e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cb2f53adb77a2b3ae687dc6d6f3ab6e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b3e70c8dbee4e7e6242b31149d84bb54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_692eb51a1b03665b520f0b424ab1b579
    def get_inputs(self):
        return [
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b3e70c8dbee4e7e6242b31149d84bb54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_692eb51a1b03665b520f0b424ab1b579
    def get_inputs(self):
        return [
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cb2f53adb77a2b3ae687dc6d6f3ab6e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cb2f53adb77a2b3ae687dc6d6f3ab6e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e816145865279750ea24e473cf346779(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6069, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 6069, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9ccf4ea2f22a5c8c8153937343c3757d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e816145865279750ea24e473cf346779
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 6069, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d8394585742f6b0e295abbb86b1425e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e485e8990d1a5ae2479706aefca977fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_9338121665dabf5961b040759d84cfa3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 48, 48, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 48, 48, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ca90081b92f11a2f8581bf3091cd674c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9338121665dabf5961b040759d84cfa3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ca90081b92f11a2f8581bf3091cd674c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9338121665dabf5961b040759d84cfa3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ca90081b92f11a2f8581bf3091cd674c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9338121665dabf5961b040759d84cfa3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ca90081b92f11a2f8581bf3091cd674c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9338121665dabf5961b040759d84cfa3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ca90081b92f11a2f8581bf3091cd674c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9338121665dabf5961b040759d84cfa3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ca90081b92f11a2f8581bf3091cd674c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9338121665dabf5961b040759d84cfa3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_37f056d264e03932bb71bf284580f07b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 240], dtype='float32'),
            paddle.static.InputSpec(shape=[171, 240], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ccd951fd3e27adecd82cd1fe0b1352f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37f056d264e03932bb71bf284580f07b
    def get_inputs(self):
        return [
            paddle.uniform([171, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([171, 240], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6d94d45888e5de3e09ab9696b2370b39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce23a53a29ef92e6eec8d8a549a7d85f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_c06e5376492d5e37f49c69c5b9f2e23b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 240], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 240], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2138a30a4566037c962f15728aa5ff03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c06e5376492d5e37f49c69c5b9f2e23b
    def get_inputs(self):
        return [
            paddle.uniform([10, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 240], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_dcc194bafa335579cd06d5226fabda59(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 144, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b3e98aa7e36479d8ec98a24dd202d727(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dcc194bafa335579cd06d5226fabda59
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c291ff260df6a6c5d2b3fde3b54bce2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24b609e55c8ed8004a7da1a199591925
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e6693535d43d87ffd0285ad3e48035fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55abc8d01f55954b67368efa6cc544b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_4dae751104e86bc9ed7b5a49668b3600(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[9, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[9, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1c44d58a18be276a69b816a702f894ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4dae751104e86bc9ed7b5a49668b3600
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.26340150833129883], [0.46034830808639526], [0.16207528114318848], [-0.08197793364524841], [-0.4799332618713379], [-0.08537876605987549], [0.2270095944404602], [-0.21020689606666565], [-0.432447612285614]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.30383044481277466], [-0.0067379772663116455], [-0.3218809962272644], [0.3049466609954834], [0.342303991317749], [0.4855678081512451], [-0.052667051553726196], [-0.4440675973892212], [0.3661329746246338]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_68b2936c810774725cee3119b2f9fb8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4dae751104e86bc9ed7b5a49668b3600
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.07887744903564453], [0.31540924310684204], [-0.2665252685546875], [-0.12849152088165283], [0.4635170102119446], [-0.1288277506828308], [-0.4158199727535248], [0.08123522996902466], [-0.3419676125049591]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.009119778871536255], [0.35780632495880127], [-0.07009202241897583], [0.13681113719940186], [-0.11918747425079346], [-0.4604017734527588], [0.17320948839187622], [-0.48267507553100586], [-0.1678488850593567]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_52be11cf449842f6b0259dc1fa37bf7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4dae751104e86bc9ed7b5a49668b3600
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.17345154285430908], [-0.4735841453075409], [-0.5550603866577148], [-0.4251040518283844], [-0.9014554023742676], [-0.536878228187561], [0.2378315031528473], [-0.77287757396698], [-0.7646393775939941]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_34b70df4b85e8a9b0ef8b89d830d6969(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4dae751104e86bc9ed7b5a49668b3600
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1000869870185852], [0.10020291805267334], [-0.45769524574279785], [-0.30565184354782104], [-0.31859248876571655], [-0.8693258166313171], [-0.8254417181015015], [-0.48135021328926086], [-0.3163154125213623]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_dbded24eaebb4c890b2dcde9e189cc61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4dae751104e86bc9ed7b5a49668b3600
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4368530511856079], [0.46684616804122925], [0.23317939043045044], [0.343126118183136], [0.37048572301864624], [0.12745672464370728], [-0.2904985547065735], [0.3288099765777588], [0.3160330057144165]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.2951025366783142], [0.1575915813446045], [0.18225038051605225], [0.010741472244262695], [0.4215221405029297], [0.45149946212768555], [-0.3987027704715729], [-0.3669936954975128], [0.3321917653083801]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_5200ebb76b0628f3ecc38999affa75fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4dae751104e86bc9ed7b5a49668b3600
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.3653436005115509], [-0.4201619625091553], [-0.20712703466415405], [0.1771603226661682], [-0.06482309103012085], [0.40892404317855835], [0.18776559829711914], [-0.0013248622417449951], [-0.025652199983596802]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.10920676589012146], [0.2152063250541687], [0.19116997718811035], [0.02454829216003418], [0.1994050145149231], [-0.48527729511260986], [0.4096217751502991], [-0.12472957372665405], [-0.4282859265804291]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_a321a9c2926e9f13265fb8e9a7c2bb02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8536edfb5dd3539b3c23bcac2ba62069
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b3e98aa7e36479d8ec98a24dd202d727(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dcc194bafa335579cd06d5226fabda59
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_25bacded1ecad1d067e3f97121f1bcfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de0b9afc4a25723de28c49a95ea6682e
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_652e9b65f6217f468c2017e65a56049e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45f5cf2c205f6ee7828e9c2334b2f164
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e6693535d43d87ffd0285ad3e48035fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55abc8d01f55954b67368efa6cc544b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_28bf34fc164d18118348246f38d5aa8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5feed18fa2e71b8e3dabb3b5beb0932
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.44863361120224]], [[-0.4261649250984192]], [[-0.19829675555229187]], [[0.01481938362121582]], [[0.43206214904785156]], [[0.2988715171813965]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[4.135169982910156]], [[4.135169982910156]], [[4.135169982910156]], [[4.135169982910156]], [[4.135169982910156]], [[4.135169982910156]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_cf887f402a2f49136bc1ced9444f50e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be5ae9c4245c383ff6a73071a8803822
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2f99997c92443c919b5477a750521337(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac220c649dd3c8ac3e434814385a8e3
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7f5fb00860ce1e36fe2575b4ab8414fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9631aec128c9d55dc0e417cdd91613fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_824452bb148158ce59a7138c4286fd30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5486865b270c0886eb6434b61235e938
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_bce6536d828a4e5a48f54c585023d543(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bce6536d828a4e5a48f54c585023d543(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9ecacf3b7cf28fcc121344a10f00b914(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_692eb51a1b03665b520f0b424ab1b579
    def get_inputs(self):
        return [
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9ecacf3b7cf28fcc121344a10f00b914(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_692eb51a1b03665b520f0b424ab1b579
    def get_inputs(self):
        return [
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bce6536d828a4e5a48f54c585023d543(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bce6536d828a4e5a48f54c585023d543(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b18b05307e30de4e0b0d8ff0ac8dc3e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8837995068d85ed2037c87dda63d6553
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4116, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2b1576864577ae0be06e1f7b97aa4523(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_423e2ebb1b58e32bb30759dd62f4168e
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_824452bb148158ce59a7138c4286fd30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5486865b270c0886eb6434b61235e938
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_307ec8e943a596446e9d5cf788cf8560(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdba59fb7e86bd9a3f073cbfd82e0b47
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_307ec8e943a596446e9d5cf788cf8560(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdba59fb7e86bd9a3f073cbfd82e0b47
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_307ec8e943a596446e9d5cf788cf8560(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdba59fb7e86bd9a3f073cbfd82e0b47
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_307ec8e943a596446e9d5cf788cf8560(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdba59fb7e86bd9a3f073cbfd82e0b47
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_307ec8e943a596446e9d5cf788cf8560(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdba59fb7e86bd9a3f073cbfd82e0b47
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_307ec8e943a596446e9d5cf788cf8560(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdba59fb7e86bd9a3f073cbfd82e0b47
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_324d4f1c3a79ef077ee62b5f7ac1d2bd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 288, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b26599ad0c6ec23f420f54cf817e3a35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_324d4f1c3a79ef077ee62b5f7ac1d2bd
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_cd502a4ab22691e7a36da5067b3f8ed7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b812e843f25f9f629b3ed8ee38ba590d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cd502a4ab22691e7a36da5067b3f8ed7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b812e843f25f9f629b3ed8ee38ba590d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cd502a4ab22691e7a36da5067b3f8ed7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b812e843f25f9f629b3ed8ee38ba590d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cd502a4ab22691e7a36da5067b3f8ed7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b812e843f25f9f629b3ed8ee38ba590d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cd502a4ab22691e7a36da5067b3f8ed7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b812e843f25f9f629b3ed8ee38ba590d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cd502a4ab22691e7a36da5067b3f8ed7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b812e843f25f9f629b3ed8ee38ba590d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7f5fb00860ce1e36fe2575b4ab8414fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9631aec128c9d55dc0e417cdd91613fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_879f6a4319440687426b9e356ad5e013(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 38, 38, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 38, 38, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6d7642efe3144200d20473bfe23b3b26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_879f6a4319440687426b9e356ad5e013
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6d7642efe3144200d20473bfe23b3b26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_879f6a4319440687426b9e356ad5e013
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6d7642efe3144200d20473bfe23b3b26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_879f6a4319440687426b9e356ad5e013
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6d7642efe3144200d20473bfe23b3b26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_879f6a4319440687426b9e356ad5e013
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6d7642efe3144200d20473bfe23b3b26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_879f6a4319440687426b9e356ad5e013
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6d7642efe3144200d20473bfe23b3b26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_879f6a4319440687426b9e356ad5e013
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6d94d45888e5de3e09ab9696b2370b39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce23a53a29ef92e6eec8d8a549a7d85f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e6693535d43d87ffd0285ad3e48035fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55abc8d01f55954b67368efa6cc544b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_1df01a60f25e84fe7f62858a629f443f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 22, 22, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 22, 22, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b5f8c386d8c308ba159343d4bf5fa4d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1df01a60f25e84fe7f62858a629f443f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b5f8c386d8c308ba159343d4bf5fa4d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1df01a60f25e84fe7f62858a629f443f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b5f8c386d8c308ba159343d4bf5fa4d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1df01a60f25e84fe7f62858a629f443f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b5f8c386d8c308ba159343d4bf5fa4d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1df01a60f25e84fe7f62858a629f443f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b5f8c386d8c308ba159343d4bf5fa4d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1df01a60f25e84fe7f62858a629f443f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b5f8c386d8c308ba159343d4bf5fa4d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1df01a60f25e84fe7f62858a629f443f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_655b68e0dc53d0b72ddec8a9d8e395e4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 480], dtype='float32'),
            paddle.static.InputSpec(shape=[145, 480], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9a7bcda0e1de41b215933c061c4bfc7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655b68e0dc53d0b72ddec8a9d8e395e4
    def get_inputs(self):
        return [
            paddle.uniform([145, 480], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([145, 480], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c291ff260df6a6c5d2b3fde3b54bce2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24b609e55c8ed8004a7da1a199591925
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e6693535d43d87ffd0285ad3e48035fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55abc8d01f55954b67368efa6cc544b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b26599ad0c6ec23f420f54cf817e3a35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_324d4f1c3a79ef077ee62b5f7ac1d2bd
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_549018118c774c5ea626225dd22333f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_569c7b46c0422c9cc58e3eb62805439e
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2f99997c92443c919b5477a750521337(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac220c649dd3c8ac3e434814385a8e3
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_5093af9f2e64730414e72e7272e86f96(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_92e89fd237307bc9e7ecd5aa36458f58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5093af9f2e64730414e72e7272e86f96
    def get_inputs(self):
        return [
            paddle.uniform([10, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2f99997c92443c919b5477a750521337(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac220c649dd3c8ac3e434814385a8e3
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f94e0924fcf405cac7f78c21754fd5d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228fdc912558e98bd21a04f8a2a84977
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_25bacded1ecad1d067e3f97121f1bcfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de0b9afc4a25723de28c49a95ea6682e
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2f99997c92443c919b5477a750521337(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac220c649dd3c8ac3e434814385a8e3
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_7c154cf805a8687f6943f81c309b8279(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 96, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_16169e72246e13019c5870d336c28b6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c154cf805a8687f6943f81c309b8279
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_16169e72246e13019c5870d336c28b6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c154cf805a8687f6943f81c309b8279
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_16169e72246e13019c5870d336c28b6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c154cf805a8687f6943f81c309b8279
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_16169e72246e13019c5870d336c28b6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c154cf805a8687f6943f81c309b8279
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_9bde1055c6b02c0b8aa3f56ba7cb21e0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 24, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_79e074d012cbe358b1857ca714310b85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9bde1055c6b02c0b8aa3f56ba7cb21e0
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_79e074d012cbe358b1857ca714310b85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9bde1055c6b02c0b8aa3f56ba7cb21e0
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_79e074d012cbe358b1857ca714310b85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9bde1055c6b02c0b8aa3f56ba7cb21e0
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_79e074d012cbe358b1857ca714310b85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9bde1055c6b02c0b8aa3f56ba7cb21e0
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b26599ad0c6ec23f420f54cf817e3a35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_324d4f1c3a79ef077ee62b5f7ac1d2bd
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_652e9b65f6217f468c2017e65a56049e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45f5cf2c205f6ee7828e9c2334b2f164
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_549018118c774c5ea626225dd22333f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_569c7b46c0422c9cc58e3eb62805439e
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_25bacded1ecad1d067e3f97121f1bcfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de0b9afc4a25723de28c49a95ea6682e
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6d94d45888e5de3e09ab9696b2370b39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce23a53a29ef92e6eec8d8a549a7d85f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_25bacded1ecad1d067e3f97121f1bcfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de0b9afc4a25723de28c49a95ea6682e
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_1c8282bfc8d9625c71eb37c7566f5842(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 44, 44, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 44, 44, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_40698ef798a36a46126ef6e37173523e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c8282bfc8d9625c71eb37c7566f5842
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_40698ef798a36a46126ef6e37173523e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c8282bfc8d9625c71eb37c7566f5842
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_40698ef798a36a46126ef6e37173523e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c8282bfc8d9625c71eb37c7566f5842
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_40698ef798a36a46126ef6e37173523e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c8282bfc8d9625c71eb37c7566f5842
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_40698ef798a36a46126ef6e37173523e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c8282bfc8d9625c71eb37c7566f5842
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_40698ef798a36a46126ef6e37173523e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c8282bfc8d9625c71eb37c7566f5842
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c310cce4553ed7afb0f29ab5043cbdb8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 11, 11, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 11, 11, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3f576f4f82a8fc3df8659e35a7cf6bd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c310cce4553ed7afb0f29ab5043cbdb8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3f576f4f82a8fc3df8659e35a7cf6bd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c310cce4553ed7afb0f29ab5043cbdb8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3f576f4f82a8fc3df8659e35a7cf6bd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c310cce4553ed7afb0f29ab5043cbdb8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3f576f4f82a8fc3df8659e35a7cf6bd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c310cce4553ed7afb0f29ab5043cbdb8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3f576f4f82a8fc3df8659e35a7cf6bd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c310cce4553ed7afb0f29ab5043cbdb8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3f576f4f82a8fc3df8659e35a7cf6bd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c310cce4553ed7afb0f29ab5043cbdb8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_65a393bdcb772f4c946958199883a04a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_852b63e07076200b2dd5a3d370d7746e
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_cf887f402a2f49136bc1ced9444f50e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be5ae9c4245c383ff6a73071a8803822
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_e3d8064caf8994bd1d0144f8d378469a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_181820ba7450e6752b702c305ae846b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3d8064caf8994bd1d0144f8d378469a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.009030580520629883]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.49611642956733704]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_2051e2b0599ae87798cd6ef2f6f2be25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3d8064caf8994bd1d0144f8d378469a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.17337018251419067]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.13599738478660583]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_6a7c7be92a232c983aa8d319d78e0df0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3d8064caf8994bd1d0144f8d378469a
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.7297652959823608]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[3.402820018375656e+38]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_0242ed15d16d8212efc7638059b2c41f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3d8064caf8994bd1d0144f8d378469a
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.23912909626960754]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[3.402820018375656e+38]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_38a54803c473dbfac2b2f0e46f46da12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3d8064caf8994bd1d0144f8d378469a
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.4661974608898163]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.23364883661270142]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_caa96eebbb57ab16fd85b208b6cbc7a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3d8064caf8994bd1d0144f8d378469a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.10313171148300171]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.22258034348487854]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_cf887f402a2f49136bc1ced9444f50e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be5ae9c4245c383ff6a73071a8803822
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e6693535d43d87ffd0285ad3e48035fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55abc8d01f55954b67368efa6cc544b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3f576f4f82a8fc3df8659e35a7cf6bd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c310cce4553ed7afb0f29ab5043cbdb8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3f576f4f82a8fc3df8659e35a7cf6bd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c310cce4553ed7afb0f29ab5043cbdb8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3f576f4f82a8fc3df8659e35a7cf6bd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c310cce4553ed7afb0f29ab5043cbdb8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3f576f4f82a8fc3df8659e35a7cf6bd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c310cce4553ed7afb0f29ab5043cbdb8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3f576f4f82a8fc3df8659e35a7cf6bd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c310cce4553ed7afb0f29ab5043cbdb8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3f576f4f82a8fc3df8659e35a7cf6bd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c310cce4553ed7afb0f29ab5043cbdb8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7f5fb00860ce1e36fe2575b4ab8414fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9631aec128c9d55dc0e417cdd91613fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_574a90fd88bca8631187ac8eac5483b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b629aeb02983c44eb0a04a805e9a408d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_574a90fd88bca8631187ac8eac5483b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b629aeb02983c44eb0a04a805e9a408d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_574a90fd88bca8631187ac8eac5483b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b629aeb02983c44eb0a04a805e9a408d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_574a90fd88bca8631187ac8eac5483b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b629aeb02983c44eb0a04a805e9a408d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_574a90fd88bca8631187ac8eac5483b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b629aeb02983c44eb0a04a805e9a408d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_574a90fd88bca8631187ac8eac5483b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b629aeb02983c44eb0a04a805e9a408d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b26599ad0c6ec23f420f54cf817e3a35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_324d4f1c3a79ef077ee62b5f7ac1d2bd
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d8394585742f6b0e295abbb86b1425e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e485e8990d1a5ae2479706aefca977fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_25bacded1ecad1d067e3f97121f1bcfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de0b9afc4a25723de28c49a95ea6682e
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e6693535d43d87ffd0285ad3e48035fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55abc8d01f55954b67368efa6cc544b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8c07ea6383662793c01e8d4a939dac4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8612d0d68b8a650001a86691bd316818
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8c07ea6383662793c01e8d4a939dac4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8612d0d68b8a650001a86691bd316818
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8c07ea6383662793c01e8d4a939dac4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8612d0d68b8a650001a86691bd316818
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8c07ea6383662793c01e8d4a939dac4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8612d0d68b8a650001a86691bd316818
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8c07ea6383662793c01e8d4a939dac4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8612d0d68b8a650001a86691bd316818
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8c07ea6383662793c01e8d4a939dac4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8612d0d68b8a650001a86691bd316818
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f0b59847981519d8ac4ef5e1b2e87404(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f0b59847981519d8ac4ef5e1b2e87404(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_837277583d6bbe1a22aaff24187d5e61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_692eb51a1b03665b520f0b424ab1b579
    def get_inputs(self):
        return [
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_837277583d6bbe1a22aaff24187d5e61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_692eb51a1b03665b520f0b424ab1b579
    def get_inputs(self):
        return [
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f0b59847981519d8ac4ef5e1b2e87404(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f0b59847981519d8ac4ef5e1b2e87404(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_aa34871db374acdec618a38f76748a35(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 11109, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 11109, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d9a688195f19799e7e3b72dba08675c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa34871db374acdec618a38f76748a35
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 11109, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_65a393bdcb772f4c946958199883a04a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_852b63e07076200b2dd5a3d370d7746e
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2f99997c92443c919b5477a750521337(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac220c649dd3c8ac3e434814385a8e3
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4c700927289cf3a5120debd41da1e8c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a860fac71210374cca5ca09f1371af03
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4c700927289cf3a5120debd41da1e8c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a860fac71210374cca5ca09f1371af03
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4c700927289cf3a5120debd41da1e8c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a860fac71210374cca5ca09f1371af03
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4c700927289cf3a5120debd41da1e8c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a860fac71210374cca5ca09f1371af03
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4c700927289cf3a5120debd41da1e8c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a860fac71210374cca5ca09f1371af03
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4c700927289cf3a5120debd41da1e8c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a860fac71210374cca5ca09f1371af03
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2f99997c92443c919b5477a750521337(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac220c649dd3c8ac3e434814385a8e3
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_549018118c774c5ea626225dd22333f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_569c7b46c0422c9cc58e3eb62805439e
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_40698ef798a36a46126ef6e37173523e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c8282bfc8d9625c71eb37c7566f5842
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_40698ef798a36a46126ef6e37173523e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c8282bfc8d9625c71eb37c7566f5842
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_40698ef798a36a46126ef6e37173523e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c8282bfc8d9625c71eb37c7566f5842
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_40698ef798a36a46126ef6e37173523e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c8282bfc8d9625c71eb37c7566f5842
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_40698ef798a36a46126ef6e37173523e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c8282bfc8d9625c71eb37c7566f5842
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_40698ef798a36a46126ef6e37173523e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c8282bfc8d9625c71eb37c7566f5842
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_59d203d0a05ed9db4f7eac54c8a587da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_59d203d0a05ed9db4f7eac54c8a587da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_53fe7914d9698ea443b701a8c4546505(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_692eb51a1b03665b520f0b424ab1b579
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_53fe7914d9698ea443b701a8c4546505(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_692eb51a1b03665b520f0b424ab1b579
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_59d203d0a05ed9db4f7eac54c8a587da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_59d203d0a05ed9db4f7eac54c8a587da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_2b46d9f3353d2bfe11be89d204713eee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2100, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 2100, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cfc4a203ac82c8c923240a050fa48e48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b46d9f3353d2bfe11be89d204713eee
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 2100, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b3e98aa7e36479d8ec98a24dd202d727(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dcc194bafa335579cd06d5226fabda59
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_18d78562d215460cc83c25380db10a2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d78d8de419b03854af5c7f9bc2f0422
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_cf887f402a2f49136bc1ced9444f50e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be5ae9c4245c383ff6a73071a8803822
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4c4724c2226687e6d0aa384097b54876(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4c4724c2226687e6d0aa384097b54876(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6ed6f3d104b7a96a1435ca5e1005a703(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_692eb51a1b03665b520f0b424ab1b579
    def get_inputs(self):
        return [
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6ed6f3d104b7a96a1435ca5e1005a703(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_692eb51a1b03665b520f0b424ab1b579
    def get_inputs(self):
        return [
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4c4724c2226687e6d0aa384097b54876(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4c4724c2226687e6d0aa384097b54876(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f86d54552cdedd21652c12a9c5d77456(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d934d40258676c7c08f228ee4092704
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3549, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2f99997c92443c919b5477a750521337(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac220c649dd3c8ac3e434814385a8e3
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_30d9a2273f71450ff2f48b7dcf1a0498(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea1b9147a2ba0a1f0e44a663741d784a
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_25bacded1ecad1d067e3f97121f1bcfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de0b9afc4a25723de28c49a95ea6682e
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6efa63011d490480fd71159b54a6142a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ef48365d12968559582b82ede7e159a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6efa63011d490480fd71159b54a6142a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ef48365d12968559582b82ede7e159a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6efa63011d490480fd71159b54a6142a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ef48365d12968559582b82ede7e159a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6efa63011d490480fd71159b54a6142a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ef48365d12968559582b82ede7e159a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6efa63011d490480fd71159b54a6142a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ef48365d12968559582b82ede7e159a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6efa63011d490480fd71159b54a6142a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ef48365d12968559582b82ede7e159a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_88f101d3b83ff8bf4eb6cd4531f0e5f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61a593482a75fcd987aa4aed1be4abf9
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e6693535d43d87ffd0285ad3e48035fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55abc8d01f55954b67368efa6cc544b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e6693535d43d87ffd0285ad3e48035fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55abc8d01f55954b67368efa6cc544b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0318b881e6a6ae476f295ae0cadf46da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1934b19eec6f2d63afc5b3cb82107f16
    def get_inputs(self):
        return [
            paddle.uniform([1, 672], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 672], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc4fbe401107c1bc8a817becc87e36f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92a93e26fdabbeac29ed6b50f4b9a312
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc4fbe401107c1bc8a817becc87e36f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92a93e26fdabbeac29ed6b50f4b9a312
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc4fbe401107c1bc8a817becc87e36f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92a93e26fdabbeac29ed6b50f4b9a312
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc4fbe401107c1bc8a817becc87e36f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92a93e26fdabbeac29ed6b50f4b9a312
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc4fbe401107c1bc8a817becc87e36f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92a93e26fdabbeac29ed6b50f4b9a312
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc4fbe401107c1bc8a817becc87e36f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92a93e26fdabbeac29ed6b50f4b9a312
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6d0f035fab2bdae285a54efde0748ab0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 156], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 156], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e91b110ec1e5dc4a7a3434f3acb6f06e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d0f035fab2bdae285a54efde0748ab0
    def get_inputs(self):
        return [
            paddle.uniform([1, 156], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 156], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_824452bb148158ce59a7138c4286fd30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5486865b270c0886eb6434b61235e938
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ef5072397f1ec03676f334a2d3bb84ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb1e03bb22a924dac4f6d857df404f9
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.4468441307544708]], [[0.4903221130371094]], [[0.5349292755126953]], [[0.5423062443733215]], [[0.4022231101989746]], [[0.608171820640564]], [[0.5130035877227783]], [[0.4183555543422699]], [[0.5500997304916382]], [[0.43059056997299194]], [[0.5732475519180298]], [[0.43051764369010925]], [[0.5852348208427429]], [[0.37766554951667786]], [[0.4168500304222107]], [[0.5743144750595093]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_1692120a45656d562794f257f279b95f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5c47a198e24d413742ddc4e8f90294c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1692120a45656d562794f257f279b95f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a23af7c7fee607c7c9349718dc14a5c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0feea93e0825a4843b257e893028a9a9
    def get_inputs(self):
        return [
            paddle.uniform([145, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([145, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6d94d45888e5de3e09ab9696b2370b39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce23a53a29ef92e6eec8d8a549a7d85f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6d94d45888e5de3e09ab9696b2370b39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce23a53a29ef92e6eec8d8a549a7d85f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2b1576864577ae0be06e1f7b97aa4523(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_423e2ebb1b58e32bb30759dd62f4168e
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3e9b9c62a146fa260bf0d800de00f030(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8052a12c240ca6fb659f402a78b33975
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c291ff260df6a6c5d2b3fde3b54bce2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24b609e55c8ed8004a7da1a199591925
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c291ff260df6a6c5d2b3fde3b54bce2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24b609e55c8ed8004a7da1a199591925
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c291ff260df6a6c5d2b3fde3b54bce2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24b609e55c8ed8004a7da1a199591925
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c291ff260df6a6c5d2b3fde3b54bce2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24b609e55c8ed8004a7da1a199591925
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_a06c6af1188ca33554585de4e524cb44(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 24, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fb2284712b10c07d4d4f54207b375b48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a06c6af1188ca33554585de4e524cb44
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.42418748140335083]], [[1.2771271467208862]], [[2.4459211826324463]], [[-0.9511007070541382]], [[-1.362120509147644]], [[0.001982450485229492]], [[0.8842978477478027]], [[-1.0343555212020874]], [[-0.7261601686477661]], [[3.013796806335449]], [[-0.8052194118499756]], [[1.2891851663589478]], [[0.06433439254760742]], [[0.7106450200080872]], [[1.9865678548812866]], [[-1.0501649379730225]], [[0.41633346676826477]], [[0.9811534881591797]], [[-1.0150970220565796]], [[-0.3432319760322571]], [[1.315356969833374]], [[0.20667269825935364]], [[-0.6107468605041504]], [[1.54805588722229]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_aab170adc23033641cdc6641b2a919eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a06c6af1188ca33554585de4e524cb44
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-3.2338485717773438]], [[3.5499796867370605]], [[-3.6019740104675293]], [[-3.1069061756134033]], [[-2.658082962036133]], [[0.8967139720916748]], [[0.9470024108886719]], [[2.1101932525634766]], [[0.3660065829753876]], [[4.07158088684082]], [[0.6674299240112305]], [[1.1017467975616455]], [[-0.102855384349823]], [[2.0153696537017822]], [[3.7119526863098145]], [[-5.380918025970459]], [[-5.348049640655518]], [[-0.25527942180633545]], [[1.4361348152160645]], [[0.09769514203071594]], [[4.8822126388549805]], [[-2.016575336456299]], [[4.633398532867432]], [[4.584447383880615]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_20d6db823dddeee0088c9150b2058b0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a06c6af1188ca33554585de4e524cb44
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.17788970470428467]], [[-1.8235797882080078]], [[-2.679997205734253]], [[3.0982394218444824]], [[-0.47933429479599]], [[0.26931238174438477]], [[9.944920539855957]], [[-6.290741920471191]], [[3.0797152519226074]], [[3.2096545696258545]], [[-0.9013689756393433]], [[-0.503687858581543]], [[3.393099546432495]], [[3.572436809539795]], [[6.873041152954102]], [[1.8287321329116821]], [[5.035619735717773]], [[0.7809153199195862]], [[-0.5418663024902344]], [[-5.004980087280273]], [[3.4555697441101074]], [[-1.795461654663086]], [[0.313679039478302]], [[-0.9901344776153564]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7c17cea39822e6a83c5fc6b87c77b4e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a06c6af1188ca33554585de4e524cb44
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.712058424949646]], [[-3.0649235248565674]], [[3.280132532119751]], [[-2.5518593788146973]], [[1.8582348823547363]], [[2.2398734092712402]], [[0.15969279408454895]], [[1.3327674865722656]], [[-0.48222798109054565]], [[1.330007553100586]], [[1.6028589010238647]], [[0.1638200581073761]], [[-4.006744861602783]], [[2.651977777481079]], [[-0.18115484714508057]], [[-0.6350057125091553]], [[-0.16288256645202637]], [[-1.4942678213119507]], [[2.9971814155578613]], [[-1.3142420053482056]], [[-0.9569375514984131]], [[1.261085033416748]], [[1.0362417697906494]], [[-0.5392221212387085]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3e9b9c62a146fa260bf0d800de00f030(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8052a12c240ca6fb659f402a78b33975
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_cf887f402a2f49136bc1ced9444f50e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be5ae9c4245c383ff6a73071a8803822
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_652e9b65f6217f468c2017e65a56049e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45f5cf2c205f6ee7828e9c2334b2f164
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7f5fb00860ce1e36fe2575b4ab8414fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9631aec128c9d55dc0e417cdd91613fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c291ff260df6a6c5d2b3fde3b54bce2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24b609e55c8ed8004a7da1a199591925
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_65a393bdcb772f4c946958199883a04a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_852b63e07076200b2dd5a3d370d7746e
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0fb232b9dc0b7db8c6b2cf623180552b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b74dfbe80ca4eee035b1a8a60be23b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0fb232b9dc0b7db8c6b2cf623180552b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b74dfbe80ca4eee035b1a8a60be23b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0fb232b9dc0b7db8c6b2cf623180552b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b74dfbe80ca4eee035b1a8a60be23b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0fb232b9dc0b7db8c6b2cf623180552b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b74dfbe80ca4eee035b1a8a60be23b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0fb232b9dc0b7db8c6b2cf623180552b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b74dfbe80ca4eee035b1a8a60be23b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0fb232b9dc0b7db8c6b2cf623180552b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b74dfbe80ca4eee035b1a8a60be23b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d8394585742f6b0e295abbb86b1425e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e485e8990d1a5ae2479706aefca977fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c291ff260df6a6c5d2b3fde3b54bce2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24b609e55c8ed8004a7da1a199591925
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_cf887f402a2f49136bc1ced9444f50e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be5ae9c4245c383ff6a73071a8803822
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c291ff260df6a6c5d2b3fde3b54bce2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24b609e55c8ed8004a7da1a199591925
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_94d069087b516e3411c44030992cbfe8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 480], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 480], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_beb9b3f7ffe24436ded39616d33efa13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94d069087b516e3411c44030992cbfe8
    def get_inputs(self):
        return [
            paddle.uniform([22, 480], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 480], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6ecb4f7857168d6be86d3c5cdc8f3723(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 872], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 872], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_74fc2eab40a75fa542774d31b71e3476(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ecb4f7857168d6be86d3c5cdc8f3723
    def get_inputs(self):
        return [
            paddle.uniform([1, 872], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 872], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0342f6fdbef59c291fde2cbee699f1b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fdfc90c9e2c9a224a34d7c28b1a71a8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ca90081b92f11a2f8581bf3091cd674c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9338121665dabf5961b040759d84cfa3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ca90081b92f11a2f8581bf3091cd674c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9338121665dabf5961b040759d84cfa3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ca90081b92f11a2f8581bf3091cd674c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9338121665dabf5961b040759d84cfa3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ca90081b92f11a2f8581bf3091cd674c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9338121665dabf5961b040759d84cfa3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ca90081b92f11a2f8581bf3091cd674c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9338121665dabf5961b040759d84cfa3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ca90081b92f11a2f8581bf3091cd674c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9338121665dabf5961b040759d84cfa3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_66be25e13d2819d0e703c5c4a5fffa3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8939003db30e283086fd87b133983
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b3e98aa7e36479d8ec98a24dd202d727(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dcc194bafa335579cd06d5226fabda59
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f94e0924fcf405cac7f78c21754fd5d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228fdc912558e98bd21a04f8a2a84977
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c291ff260df6a6c5d2b3fde3b54bce2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24b609e55c8ed8004a7da1a199591925
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_338f4762f367f6e85f71b245f764c715(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3213aced3839e6653017d5749c904e32
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_338f4762f367f6e85f71b245f764c715(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3213aced3839e6653017d5749c904e32
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_338f4762f367f6e85f71b245f764c715(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3213aced3839e6653017d5749c904e32
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_338f4762f367f6e85f71b245f764c715(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3213aced3839e6653017d5749c904e32
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_338f4762f367f6e85f71b245f764c715(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3213aced3839e6653017d5749c904e32
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_338f4762f367f6e85f71b245f764c715(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3213aced3839e6653017d5749c904e32
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_74fc2eab40a75fa542774d31b71e3476(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ecb4f7857168d6be86d3c5cdc8f3723
    def get_inputs(self):
        return [
            paddle.uniform([1, 872], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 872], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8168375b2b674f5b7119726d60ad87c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8168375b2b674f5b7119726d60ad87c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4f6b7f2751ec4c78483237d3e2b515ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_692eb51a1b03665b520f0b424ab1b579
    def get_inputs(self):
        return [
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4f6b7f2751ec4c78483237d3e2b515ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_692eb51a1b03665b520f0b424ab1b579
    def get_inputs(self):
        return [
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8168375b2b674f5b7119726d60ad87c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8168375b2b674f5b7119726d60ad87c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7aa39faadd38863893499ff1330c086b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8400, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8400, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_22ce95b7d922fba342feb5f1fca11b77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7aa39faadd38863893499ff1330c086b
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8400, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_30d9a2273f71450ff2f48b7dcf1a0498(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea1b9147a2ba0a1f0e44a663741d784a
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e6693535d43d87ffd0285ad3e48035fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55abc8d01f55954b67368efa6cc544b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c291ff260df6a6c5d2b3fde3b54bce2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24b609e55c8ed8004a7da1a199591925
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6d7642efe3144200d20473bfe23b3b26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_879f6a4319440687426b9e356ad5e013
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6d7642efe3144200d20473bfe23b3b26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_879f6a4319440687426b9e356ad5e013
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6d7642efe3144200d20473bfe23b3b26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_879f6a4319440687426b9e356ad5e013
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6d7642efe3144200d20473bfe23b3b26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_879f6a4319440687426b9e356ad5e013
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6d7642efe3144200d20473bfe23b3b26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_879f6a4319440687426b9e356ad5e013
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6d7642efe3144200d20473bfe23b3b26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_879f6a4319440687426b9e356ad5e013
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2f99997c92443c919b5477a750521337(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac220c649dd3c8ac3e434814385a8e3
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_25bacded1ecad1d067e3f97121f1bcfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de0b9afc4a25723de28c49a95ea6682e
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6d94d45888e5de3e09ab9696b2370b39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce23a53a29ef92e6eec8d8a549a7d85f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7f5fb00860ce1e36fe2575b4ab8414fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9631aec128c9d55dc0e417cdd91613fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f94e0924fcf405cac7f78c21754fd5d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228fdc912558e98bd21a04f8a2a84977
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_40423bc823aaec28fa31a2b3be882cbb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 92], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7e021fd99a0a6425f1f5e8a50f06eef8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40423bc823aaec28fa31a2b3be882cbb
    def get_inputs(self):
        return [
            paddle.uniform([1, 92], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_30d9a2273f71450ff2f48b7dcf1a0498(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea1b9147a2ba0a1f0e44a663741d784a
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b3e98aa7e36479d8ec98a24dd202d727(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dcc194bafa335579cd06d5226fabda59
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_658d985704bc06937f9c6471330dca45(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8b41ee56529ad98769f37c2d76116aac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_658d985704bc06937f9c6471330dca45
    def get_inputs(self):
        return [
            paddle.uniform([22, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2f99997c92443c919b5477a750521337(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac220c649dd3c8ac3e434814385a8e3
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6d94d45888e5de3e09ab9696b2370b39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce23a53a29ef92e6eec8d8a549a7d85f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_b96f628d783ca034f89835dc5764a205(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 40, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_570876076fbdd459811154aa8d481387(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b96f628d783ca034f89835dc5764a205
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a321a9c2926e9f13265fb8e9a7c2bb02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8536edfb5dd3539b3c23bcac2ba62069
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6d94d45888e5de3e09ab9696b2370b39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce23a53a29ef92e6eec8d8a549a7d85f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_90809ede70d0263cf7e9eb63ced4c32f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_90809ede70d0263cf7e9eb63ced4c32f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_850015d5d228a895a5e9eeeaad69afbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_692eb51a1b03665b520f0b424ab1b579
    def get_inputs(self):
        return [
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_850015d5d228a895a5e9eeeaad69afbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_692eb51a1b03665b520f0b424ab1b579
    def get_inputs(self):
        return [
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_90809ede70d0263cf7e9eb63ced4c32f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_90809ede70d0263cf7e9eb63ced4c32f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_acbf832a4d10fce50a3bbb720fcd093e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9261, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9261, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ec288a3a616e0eb6273f488a2a7a0d4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_acbf832a4d10fce50a3bbb720fcd093e
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9261, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f21ee850fe42b0dc3de70a05d66d91d1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[171, 36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_33d3a58919bb307c021fbd9b6ac679cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f21ee850fe42b0dc3de70a05d66d91d1
    def get_inputs(self):
        return [
            paddle.uniform([171, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([171, 36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3af73b6b1ba18850ef9621a2ad8ada4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3af73b6b1ba18850ef9621a2ad8ada4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_74b640e82f7698ab3e0ee3fee058ad97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_692eb51a1b03665b520f0b424ab1b579
    def get_inputs(self):
        return [
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_74b640e82f7698ab3e0ee3fee058ad97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_692eb51a1b03665b520f0b424ab1b579
    def get_inputs(self):
        return [
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3af73b6b1ba18850ef9621a2ad8ada4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3af73b6b1ba18850ef9621a2ad8ada4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d1149fc9572fc54b986cd61768768803(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 7581, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 7581, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c0fffa0543cfb7a83defef31bdaa3e36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1149fc9572fc54b986cd61768768803
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 7581, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_68912ca72501a4e3e4d0d5e856b30366(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1248], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1248], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4fd91edafe0c6531e3f8d6bc0cd06403(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68912ca72501a4e3e4d0d5e856b30366
    def get_inputs(self):
        return [
            paddle.uniform([1, 1248], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1248], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_549018118c774c5ea626225dd22333f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_569c7b46c0422c9cc58e3eb62805439e
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_3518265685cdff22b9d428de095c9927(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 120], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 120], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f6f0a368976a815c2836174588c1556d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3518265685cdff22b9d428de095c9927
    def get_inputs(self):
        return [
            paddle.uniform([1, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 120], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b5f8c386d8c308ba159343d4bf5fa4d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1df01a60f25e84fe7f62858a629f443f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b5f8c386d8c308ba159343d4bf5fa4d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1df01a60f25e84fe7f62858a629f443f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b5f8c386d8c308ba159343d4bf5fa4d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1df01a60f25e84fe7f62858a629f443f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b5f8c386d8c308ba159343d4bf5fa4d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1df01a60f25e84fe7f62858a629f443f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b5f8c386d8c308ba159343d4bf5fa4d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1df01a60f25e84fe7f62858a629f443f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b5f8c386d8c308ba159343d4bf5fa4d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1df01a60f25e84fe7f62858a629f443f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a461680d38504b648a1f24890fb9d0b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1ee98d99e145aefac757d4fb583eb4a
    def get_inputs(self):
        return [
            paddle.uniform([22, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0b5ad9b6a6ecb71041394d21930b720a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 240], dtype='float32'),
            paddle.static.InputSpec(shape=[145, 240], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a21f980a2ac7c2a52e9ae46a572d626e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b5ad9b6a6ecb71041394d21930b720a
    def get_inputs(self):
        return [
            paddle.uniform([145, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([145, 240], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6d94d45888e5de3e09ab9696b2370b39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce23a53a29ef92e6eec8d8a549a7d85f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_87197829c62ddf2f92712fec38a23a87(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 336], dtype='float32'),
            paddle.static.InputSpec(shape=[171, 336], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e5fcc6bc8d3c028c5f9cadce5b16b625(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87197829c62ddf2f92712fec38a23a87
    def get_inputs(self):
        return [
            paddle.uniform([171, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([171, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a9eb66426d5a7093834e5841f44120c9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 32, 100, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 32, 100, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6b5e65937317d44e1a8eec9cbac2a682(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9eb66426d5a7093834e5841f44120c9
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 100, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 32, 100, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6d94d45888e5de3e09ab9696b2370b39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce23a53a29ef92e6eec8d8a549a7d85f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_25bacded1ecad1d067e3f97121f1bcfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de0b9afc4a25723de28c49a95ea6682e
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2f99997c92443c919b5477a750521337(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac220c649dd3c8ac3e434814385a8e3
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7f5fb00860ce1e36fe2575b4ab8414fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9631aec128c9d55dc0e417cdd91613fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e5fcc6bc8d3c028c5f9cadce5b16b625(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87197829c62ddf2f92712fec38a23a87
    def get_inputs(self):
        return [
            paddle.uniform([171, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([171, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2f99997c92443c919b5477a750521337(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac220c649dd3c8ac3e434814385a8e3
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f263286b03f2f30037f5f269fbeee3f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2850aeb421d78f007bb2e5ce26a7c30c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f263286b03f2f30037f5f269fbeee3f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2850aeb421d78f007bb2e5ce26a7c30c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f263286b03f2f30037f5f269fbeee3f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2850aeb421d78f007bb2e5ce26a7c30c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f263286b03f2f30037f5f269fbeee3f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2850aeb421d78f007bb2e5ce26a7c30c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f263286b03f2f30037f5f269fbeee3f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2850aeb421d78f007bb2e5ce26a7c30c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f263286b03f2f30037f5f269fbeee3f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2850aeb421d78f007bb2e5ce26a7c30c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_88f101d3b83ff8bf4eb6cd4531f0e5f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61a593482a75fcd987aa4aed1be4abf9
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_09c85685f7f3b4e5a77689eea932af4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce18fb8c6d189e028ab68300eb069935
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_66be25e13d2819d0e703c5c4a5fffa3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8939003db30e283086fd87b133983
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_037bd1a601caa8d73929900fba5bd5ce(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 480], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 480], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_658b45f75a972997ec799355a8be9407(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_037bd1a601caa8d73929900fba5bd5ce
    def get_inputs(self):
        return [
            paddle.uniform([10, 480], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 480], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c291ff260df6a6c5d2b3fde3b54bce2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24b609e55c8ed8004a7da1a199591925
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e6693535d43d87ffd0285ad3e48035fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55abc8d01f55954b67368efa6cc544b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b26599ad0c6ec23f420f54cf817e3a35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_324d4f1c3a79ef077ee62b5f7ac1d2bd
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3e9b9c62a146fa260bf0d800de00f030(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8052a12c240ca6fb659f402a78b33975
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5c47a198e24d413742ddc4e8f90294c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1692120a45656d562794f257f279b95f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d8394585742f6b0e295abbb86b1425e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e485e8990d1a5ae2479706aefca977fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]




if __name__ == '__main__':
    unittest.main()