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


class PrimitiveOp_254443cd513f42109ee8cb1ce75a1e34(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1542, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1542, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_47bc4403bd63e6da288f6af2ec520555(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_254443cd513f42109ee8cb1ce75a1e34
    def get_inputs(self):
        return [
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_47bc4403bd63e6da288f6af2ec520555(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_254443cd513f42109ee8cb1ce75a1e34
    def get_inputs(self):
        return [
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_47bc4403bd63e6da288f6af2ec520555(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_254443cd513f42109ee8cb1ce75a1e34
    def get_inputs(self):
        return [
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_47bc4403bd63e6da288f6af2ec520555(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_254443cd513f42109ee8cb1ce75a1e34
    def get_inputs(self):
        return [
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_47bc4403bd63e6da288f6af2ec520555(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_254443cd513f42109ee8cb1ce75a1e34
    def get_inputs(self):
        return [
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_47bc4403bd63e6da288f6af2ec520555(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_254443cd513f42109ee8cb1ce75a1e34
    def get_inputs(self):
        return [
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
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


class PrimitiveOp_9a6b84ee13a5302cf0e2800d740095e7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2361, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2361, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5621b5de54dbf49a86b944a3f2791274(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a6b84ee13a5302cf0e2800d740095e7
    def get_inputs(self):
        return [
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5621b5de54dbf49a86b944a3f2791274(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a6b84ee13a5302cf0e2800d740095e7
    def get_inputs(self):
        return [
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5621b5de54dbf49a86b944a3f2791274(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a6b84ee13a5302cf0e2800d740095e7
    def get_inputs(self):
        return [
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5621b5de54dbf49a86b944a3f2791274(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a6b84ee13a5302cf0e2800d740095e7
    def get_inputs(self):
        return [
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5621b5de54dbf49a86b944a3f2791274(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a6b84ee13a5302cf0e2800d740095e7
    def get_inputs(self):
        return [
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5621b5de54dbf49a86b944a3f2791274(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a6b84ee13a5302cf0e2800d740095e7
    def get_inputs(self):
        return [
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_bb169e0018698e29c2b2625f3ddf5bb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea88ac253d18142d28130c6adb0c2cf4
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.45028048753738403], [-0.22129765152931213], [-0.1057390570640564], [-0.43403640389442444]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.32846152782440186], [-0.13607582449913025], [-0.02927514910697937], [0.16101807355880737]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_1ac712a0c3910b7f5badc1fb95eaf026(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea88ac253d18142d28130c6adb0c2cf4
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3894405961036682], [-0.27829545736312866], [0.195406973361969], [0.47131139039993286]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.3450826406478882], [-0.3794686198234558], [-0.25792452692985535], [-0.4801425337791443]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_8cd3118869624334cee03eaa7799c995(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea88ac253d18142d28130c6adb0c2cf4
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.8135843873023987], [-0.07678523659706116], [-0.3882373571395874], [-0.4342598021030426]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_7bffe6e8189dde6766d2a7e46b1d3c6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea88ac253d18142d28130c6adb0c2cf4
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.37945646047592163], [-0.5116609334945679], [-0.05446815490722656], [-0.3714717626571655]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_a06da2c36a97f10febdaa4107929e13a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea88ac253d18142d28130c6adb0c2cf4
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.029027044773101807], [-0.14451241493225098], [0.282498300075531], [-0.09930366277694702]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.4851228594779968], [-0.2152409553527832], [0.26293325424194336], [0.00022339820861816406]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_33983590cedcda6c8104786c293a0027(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea88ac253d18142d28130c6adb0c2cf4
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.06026780605316162], [-0.3177316188812256], [-0.24746686220169067], [-0.10867077112197876]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.03437381982803345], [0.13219231367111206], [-0.20345637202262878], [-0.20608854293823242]], dtype='float32').reshape([4, 1]),
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


class TestPrimitiveOp_d868edc1f6051f8661a14ad885fcdf1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5feed18fa2e71b8e3dabb3b5beb0932
    def get_inputs(self):
        return [
            paddle.to_tensor([[[-0.4427894055843353]], [[-0.4229472279548645]], [[-0.2511310577392578]], [[0.19796502590179443]], [[-0.420658141374588]], [[-0.3204308748245239]]], dtype='float32').reshape([6, 1, 1]),
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


class TestPrimitiveOp_b9ac6e28a0878dc1a0bf9ac98bff4350(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b6f51e0ff307ce9e9ae0a6d3c1cc52e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.06431865692138672, 0.39935123920440674, 0.44849663972854614, 0.11540669202804565, 0.031954169273376465, 0.4726531505584717], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.48757103085517883, -0.2080082893371582, -0.42475807666778564, 0.46605026721954346, 0.20001959800720215, -0.3895566165447235], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_c93413e4223369827ca4b54b0a345ac7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b6f51e0ff307ce9e9ae0a6d3c1cc52e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.02541947364807129, 0.2636595368385315, 0.3311959505081177, 0.39743155241012573, -0.36470523476600647, -0.1232295036315918], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.26385489106178284, 0.4451194405555725, 0.12403160333633423, -0.2686293423175812, -0.1215813159942627, 0.21148324012756348], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_d25b0be1a75b733a90c112b013aa5fb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b6f51e0ff307ce9e9ae0a6d3c1cc52e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.06431865692138672, 0.2684096693992615, 0.29037344455718994, -0.3376345932483673, 0.031954169273376465, -0.10851988196372986], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.17642048001289368, -0.2487233579158783, -0.270376980304718, -0.13760244846343994, 0.2186264991760254, -0.2595521807670593], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_38f3d8c14d313162bc652318236542f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b6f51e0ff307ce9e9ae0a6d3c1cc52e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.02541947364807129, 0.2636595368385315, 0.3311959505081177, 0.2361714243888855, -0.43945392966270447, -0.1232295036315918], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.4874858558177948, -0.4435403347015381, 0.09530919790267944, -0.18070703744888306, 0.15153855085372925, 0.3255499601364136], dtype='float32').reshape([6]),
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


class TestPrimitiveOp_d7dcb14788555d4b756b4025e8385e11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05baf4743ceb3c4812ec461ba7942f94
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4716532230377197], [-0.4165400564670563], [-0.0014056563377380371], [0.09141331911087036], [-0.2677716016769409], [-0.08868327736854553]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.233851820230484], [0.1492787003517151], [-0.028367280960083008], [-0.4339827001094818], [-0.40483978390693665], [0.01156306266784668]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_6bf9e291021a22f3b068694021537974(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05baf4743ceb3c4812ec461ba7942f94
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.16195690631866455], [0.2852780222892761], [-0.3642190098762512], [-0.16062521934509277], [-0.4562195837497711], [-0.22439396381378174]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.47951558232307434], [0.4709428548812866], [0.08498668670654297], [0.024811983108520508], [0.25165116786956787], [0.1815364956855774]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_b8dbe7859ab7d630ce5bc242ee8ec28c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05baf4743ceb3c4812ec461ba7942f94
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.6776602268218994], [-0.8681062459945679], [-0.14143580198287964], [-0.644929051399231], [-0.7494124174118042], [-0.5479037761688232]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_6d3f2d8459f5b1fc18cd36208107f314(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05baf4743ceb3c4812ec461ba7942f94
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.1993131935596466], [-0.049627602100372314], [-0.761633574962616], [-0.5611773133277893], [-0.9184267520904541], [-0.29563701152801514]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_752744be3fc39c1861c57506f6d55159(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05baf4743ceb3c4812ec461ba7942f94
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.443808376789093], [0.451566219329834], [-0.028237223625183105], [-0.2575671076774597], [0.07678425312042236], [0.023420095443725586]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.16058611869812012], [-0.26714545488357544], [0.11306852102279663], [0.21094638109207153], [0.34457260370254517], [0.4592204689979553]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_48783e274eb8dec0b2a02b5b9af68ac5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05baf4743ceb3c4812ec461ba7942f94
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.28020238876342773], [-0.30021917819976807], [0.39741456508636475], [-0.4255662262439728], [0.46220719814300537], [-0.33687111735343933]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.4044957756996155], [0.33490562438964844], [0.13769322633743286], [0.40055209398269653], [0.0017930269241333008], [0.0712430477142334]], dtype='float32').reshape([6, 1]),
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


class TestPrimitiveOp_364cc1652c2e906921fb0b642d542a39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb1e03bb22a924dac4f6d857df404f9
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.5347432494163513]], [[0.5099735856056213]], [[0.6067728400230408]], [[0.40869805216789246]], [[0.5594559907913208]], [[0.5243954658508301]], [[0.41180869936943054]], [[0.5246237516403198]], [[0.4128502309322357]], [[0.44987213611602783]], [[0.5791157484054565]], [[0.4929652810096741]], [[0.5992297530174255]], [[0.447785347700119]], [[0.473532110452652]], [[0.44777747988700867]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class PrimitiveOp_d6a172ae67bb410d1c9739d195589a82(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2053, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2053, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_38e76d66c38ce3b98b2254298895b84f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6a172ae67bb410d1c9739d195589a82
    def get_inputs(self):
        return [
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_38e76d66c38ce3b98b2254298895b84f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6a172ae67bb410d1c9739d195589a82
    def get_inputs(self):
        return [
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_38e76d66c38ce3b98b2254298895b84f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6a172ae67bb410d1c9739d195589a82
    def get_inputs(self):
        return [
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_38e76d66c38ce3b98b2254298895b84f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6a172ae67bb410d1c9739d195589a82
    def get_inputs(self):
        return [
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_38e76d66c38ce3b98b2254298895b84f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6a172ae67bb410d1c9739d195589a82
    def get_inputs(self):
        return [
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_38e76d66c38ce3b98b2254298895b84f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6a172ae67bb410d1c9739d195589a82
    def get_inputs(self):
        return [
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_17786f970bffa2c04ee8de6a5387e920(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8e51305a7ff4a6086a9932522e3ba63
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.09874492883682251], [0.027947425842285156], [-0.41162997484207153], [-0.38247421383857727], [0.24568265676498413]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.1422344446182251], [0.3373464345932007], [0.1920177936553955], [0.2963539958000183], [-0.3881467580795288]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_a881a57a9a55925cf35b06019404a697(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8e51305a7ff4a6086a9932522e3ba63
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.05882182717323303], [0.31854957342147827], [-0.14475178718566895], [-0.1629328727722168], [0.2213781476020813]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.4251890182495117], [0.2107974886894226], [0.30673420429229736], [0.45866304636001587], [-0.3283495306968689]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_b8861d06c3cfab1e830abd2dd0562a04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8e51305a7ff4a6086a9932522e3ba63
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.4108051061630249], [0.32048124074935913], [-0.7286980152130127], [-0.7838220596313477], [-0.45773786306381226]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_e42e984e7224f7a724a3933120c9aa3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8e51305a7ff4a6086a9932522e3ba63
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.7529056072235107], [0.1669033169746399], [-0.2459903359413147], [-0.20591139793395996], [-0.47663283348083496]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_285142600e5d44c5d41fb87ce3f1f1bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8e51305a7ff4a6086a9932522e3ba63
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2685706615447998], [-0.42384102940559387], [-0.11871618032455444], [-0.3788006603717804], [0.06959110498428345]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.2158190906047821], [-0.292533814907074], [0.31706804037094116], [0.401347815990448], [0.0006369352340698242]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_39a89fb776a6f812dcd7b97fc7d8fdd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8e51305a7ff4a6086a9932522e3ba63
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.185139000415802], [0.043894171714782715], [-0.4893096685409546], [-0.23725301027297974], [0.14828330278396606]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.327716588973999], [-0.12176045775413513], [0.10123854875564575], [0.042978525161743164], [-0.41153308749198914]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_30d9a2273f71450ff2f48b7dcf1a0498(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea1b9147a2ba0a1f0e44a663741d784a
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_68102b7878cc8cdf6a7da3cb652c03d5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1825, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1825, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f9f6ed42206b982aef00f92bd5559be9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68102b7878cc8cdf6a7da3cb652c03d5
    def get_inputs(self):
        return [
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f9f6ed42206b982aef00f92bd5559be9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68102b7878cc8cdf6a7da3cb652c03d5
    def get_inputs(self):
        return [
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f9f6ed42206b982aef00f92bd5559be9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68102b7878cc8cdf6a7da3cb652c03d5
    def get_inputs(self):
        return [
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f9f6ed42206b982aef00f92bd5559be9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68102b7878cc8cdf6a7da3cb652c03d5
    def get_inputs(self):
        return [
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f9f6ed42206b982aef00f92bd5559be9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68102b7878cc8cdf6a7da3cb652c03d5
    def get_inputs(self):
        return [
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f9f6ed42206b982aef00f92bd5559be9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68102b7878cc8cdf6a7da3cb652c03d5
    def get_inputs(self):
        return [
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
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


class PrimitiveOp_22afd88a1862af53cbe7d5dceedb4b4c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3087, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[3087, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3f0248e1b958d6c3601106c91ae5a912(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22afd88a1862af53cbe7d5dceedb4b4c
    def get_inputs(self):
        return [
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3f0248e1b958d6c3601106c91ae5a912(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22afd88a1862af53cbe7d5dceedb4b4c
    def get_inputs(self):
        return [
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3f0248e1b958d6c3601106c91ae5a912(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22afd88a1862af53cbe7d5dceedb4b4c
    def get_inputs(self):
        return [
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3f0248e1b958d6c3601106c91ae5a912(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22afd88a1862af53cbe7d5dceedb4b4c
    def get_inputs(self):
        return [
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3f0248e1b958d6c3601106c91ae5a912(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22afd88a1862af53cbe7d5dceedb4b4c
    def get_inputs(self):
        return [
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3f0248e1b958d6c3601106c91ae5a912(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22afd88a1862af53cbe7d5dceedb4b4c
    def get_inputs(self):
        return [
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_b9b27e1de3d53f54e305d364c1835bb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4dae751104e86bc9ed7b5a49668b3600
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.2119961678981781], [-0.3788200914859772], [-0.04915744066238403], [0.328036367893219], [0.3646824359893799], [-0.39871159195899963], [-0.35914939641952515], [0.16614854335784912], [0.22541850805282593]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.11566966772079468], [0.32729101181030273], [-0.4864160418510437], [-0.1263948678970337], [0.4190939664840698], [0.11164069175720215], [0.20763516426086426], [-0.22272902727127075], [0.49746018648147583]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_a6bf8e53c5b3c1433b64c513781ec7b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4dae751104e86bc9ed7b5a49668b3600
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.3213467597961426], [-0.23528355360031128], [0.44690072536468506], [0.41063904762268066], [-0.278251588344574], [-0.40452665090560913], [-0.044628649950027466], [0.17216962575912476], [0.09485393762588501]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.3514004349708557], [0.19362658262252808], [0.43103843927383423], [-0.39259082078933716], [0.17742151021957397], [-0.18762415647506714], [-0.008309662342071533], [-0.009750127792358398], [-0.073464035987854]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_dcb327c30355b90ea0acd24112cfcd10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4dae751104e86bc9ed7b5a49668b3600
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.21049585938453674], [-0.7867534160614014], [-0.5523268580436707], [-0.21806353330612183], [0.5273773670196533], [-0.7422494888305664], [-0.7423455715179443], [-0.36071497201919556], [0.2993007004261017]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_51895c1544b954d72b3e193e36d19711(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4dae751104e86bc9ed7b5a49668b3600
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.4662758708000183], [-0.32748234272003174], [0.31241774559020996], [-0.12787646055221558], [-0.6326947808265686], [-0.9038877487182617], [-0.19740626215934753], [-0.10725265741348267], [-0.4795718193054199]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38], [3.402820018375656e+38]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_02718c5312cea672797d31536b764191(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4dae751104e86bc9ed7b5a49668b3600
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.1698777675628662], [0.3688810467720032], [0.06591081619262695], [0.09166866540908813], [-0.16269496083259583], [-0.0751321017742157], [-0.13586968183517456], [0.1379859447479248], [-0.4187920093536377]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.0015003085136413574], [0.4079332947731018], [-0.20244035124778748], [-0.11076605319976807], [-0.1853780746459961], [0.3435378670692444], [0.3831961750984192], [-0.0418873131275177], [-0.07388219237327576]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_60c99678ae14d59fe69313b6421c6216(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4dae751104e86bc9ed7b5a49668b3600
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.08901798725128174], [0.0880575180053711], [0.11862069368362427], [-0.3751475214958191], [0.35444319248199463], [0.4993610978126526], [-0.4423360228538513], [0.09750252962112427], [-0.12597641348838806]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.14492911100387573], [0.09219878911972046], [0.06880456209182739], [-0.2647143602371216], [0.103046715259552], [-0.24344509840011597], [0.15277761220932007], [-0.4944602847099304], [0.4061077833175659]], dtype='float32').reshape([9, 1]),
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


class TestPrimitiveOp_1d1270830d820d45245eb88e72adf90f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5feed18fa2e71b8e3dabb3b5beb0932
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.11430507898330688]], [[0.06105196475982666]], [[0.356023907661438]], [[-0.42108094692230225]], [[-0.4185469448566437]], [[-0.26130467653274536]]], dtype='float32').reshape([6, 1, 1]),
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


class PrimitiveOp_d44fa3187f6647767faf5ddb1a475fc9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2119, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2119, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d11b00569a3fed5374092a6d79430632(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d44fa3187f6647767faf5ddb1a475fc9
    def get_inputs(self):
        return [
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d11b00569a3fed5374092a6d79430632(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d44fa3187f6647767faf5ddb1a475fc9
    def get_inputs(self):
        return [
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d11b00569a3fed5374092a6d79430632(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d44fa3187f6647767faf5ddb1a475fc9
    def get_inputs(self):
        return [
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d11b00569a3fed5374092a6d79430632(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d44fa3187f6647767faf5ddb1a475fc9
    def get_inputs(self):
        return [
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d11b00569a3fed5374092a6d79430632(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d44fa3187f6647767faf5ddb1a475fc9
    def get_inputs(self):
        return [
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d11b00569a3fed5374092a6d79430632(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d44fa3187f6647767faf5ddb1a475fc9
    def get_inputs(self):
        return [
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_1af9274153d4455cbc01a9b1fa9a76a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3d8064caf8994bd1d0144f8d378469a
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.050851911306381226]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.0921517014503479]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_3e458cd6b1a6f34970144c9e46bf8482(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3d8064caf8994bd1d0144f8d378469a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3629520535469055]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.204064279794693]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_a2ac12c5cdb9dc0ac4aa0496011b3ee5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3d8064caf8994bd1d0144f8d378469a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.16702386736869812]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[3.402820018375656e+38]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_2fd8ea992e6b6bc5c8c9d8260128a10d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3d8064caf8994bd1d0144f8d378469a
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.006491541862487793]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[3.402820018375656e+38]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_353c35a2125770e8c70dde1fd7c13efa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3d8064caf8994bd1d0144f8d378469a
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.21787577867507935]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.41818875074386597]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_719e833e939e65164b93824459068304(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3d8064caf8994bd1d0144f8d378469a
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.4216543436050415]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.1975727379322052]], dtype='float32').reshape([1, 1]),
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


class PrimitiveOp_12b0331ad5bf19c87f8c2395ab0b1bd8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5606, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[5606, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_48c30ae0482b4eb1f2d3ed985e85a5ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b0331ad5bf19c87f8c2395ab0b1bd8
    def get_inputs(self):
        return [
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_48c30ae0482b4eb1f2d3ed985e85a5ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b0331ad5bf19c87f8c2395ab0b1bd8
    def get_inputs(self):
        return [
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_48c30ae0482b4eb1f2d3ed985e85a5ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b0331ad5bf19c87f8c2395ab0b1bd8
    def get_inputs(self):
        return [
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_48c30ae0482b4eb1f2d3ed985e85a5ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b0331ad5bf19c87f8c2395ab0b1bd8
    def get_inputs(self):
        return [
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_48c30ae0482b4eb1f2d3ed985e85a5ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b0331ad5bf19c87f8c2395ab0b1bd8
    def get_inputs(self):
        return [
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_48c30ae0482b4eb1f2d3ed985e85a5ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b0331ad5bf19c87f8c2395ab0b1bd8
    def get_inputs(self):
        return [
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
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


class PrimitiveOp_3d477227e677d1103bed349740c7af80(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1036, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1036, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_07c87eb6775e6e079d67ea3109aa4bf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d477227e677d1103bed349740c7af80
    def get_inputs(self):
        return [
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_07c87eb6775e6e079d67ea3109aa4bf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d477227e677d1103bed349740c7af80
    def get_inputs(self):
        return [
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_07c87eb6775e6e079d67ea3109aa4bf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d477227e677d1103bed349740c7af80
    def get_inputs(self):
        return [
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_07c87eb6775e6e079d67ea3109aa4bf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d477227e677d1103bed349740c7af80
    def get_inputs(self):
        return [
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_07c87eb6775e6e079d67ea3109aa4bf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d477227e677d1103bed349740c7af80
    def get_inputs(self):
        return [
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_07c87eb6775e6e079d67ea3109aa4bf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d477227e677d1103bed349740c7af80
    def get_inputs(self):
        return [
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
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


class PrimitiveOp_b01e4d1dc5444a210a1fc3e3e82ae465(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1809, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1809, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_de94aa2e1185ca014caca61e9af4e6b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b01e4d1dc5444a210a1fc3e3e82ae465
    def get_inputs(self):
        return [
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_de94aa2e1185ca014caca61e9af4e6b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b01e4d1dc5444a210a1fc3e3e82ae465
    def get_inputs(self):
        return [
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_de94aa2e1185ca014caca61e9af4e6b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b01e4d1dc5444a210a1fc3e3e82ae465
    def get_inputs(self):
        return [
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_de94aa2e1185ca014caca61e9af4e6b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b01e4d1dc5444a210a1fc3e3e82ae465
    def get_inputs(self):
        return [
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_de94aa2e1185ca014caca61e9af4e6b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b01e4d1dc5444a210a1fc3e3e82ae465
    def get_inputs(self):
        return [
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_de94aa2e1185ca014caca61e9af4e6b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b01e4d1dc5444a210a1fc3e3e82ae465
    def get_inputs(self):
        return [
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_6ebf975584851fea7e98e93f26da81c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb1e03bb22a924dac4f6d857df404f9
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.40343600511550903]], [[0.44983044266700745]], [[0.520755410194397]], [[0.5964234471321106]], [[0.53080815076828]], [[0.5495370626449585]], [[0.4122939705848694]], [[0.37915077805519104]], [[0.5342304706573486]], [[0.5069877505302429]], [[0.5712809562683105]], [[0.6138119101524353]], [[0.47437673807144165]], [[0.5046910643577576]], [[0.5190125703811646]], [[0.5590493679046631]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_f7627a6a8e5c8a240098f2d654936e0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a06c6af1188ca33554585de4e524cb44
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.221268653869629]], [[0.9400168657302856]], [[1.2648797035217285]], [[-0.7828768491744995]], [[0.14005321264266968]], [[-0.4859877824783325]], [[1.0602850914001465]], [[2.2105560302734375]], [[-0.014431595802307129]], [[1.0392290353775024]], [[0.8201172351837158]], [[1.365004062652588]], [[0.595476508140564]], [[-0.044568002223968506]], [[1.0066362619400024]], [[-0.8956148624420166]], [[0.8129045367240906]], [[1.548125147819519]], [[0.8165863752365112]], [[1.3588109016418457]], [[0.9757912755012512]], [[-0.03924983739852905]], [[2.208618640899658]], [[0.5072690844535828]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_41c9bb7d4ae42dedf1a0f41baecd4955(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a06c6af1188ca33554585de4e524cb44
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.432440996170044]], [[1.9133918285369873]], [[-4.841952323913574]], [[-2.986837863922119]], [[5.53218936920166]], [[0.819711446762085]], [[-2.578016996383667]], [[3.2448952198028564]], [[2.637657642364502]], [[2.5657551288604736]], [[5.677200794219971]], [[3.133399724960327]], [[-2.668001890182495]], [[6.024156093597412]], [[1.6108464002609253]], [[-4.618319988250732]], [[-1.9452579021453857]], [[5.598021984100342]], [[4.123971939086914]], [[3.6485931873321533]], [[-4.266037940979004]], [[3.7118403911590576]], [[-1.9609706401824951]], [[-0.3418116569519043]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_63dab0444a6cc1f103cea5f162e95a99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a06c6af1188ca33554585de4e524cb44
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.5313297510147095]], [[-0.7399910688400269]], [[-2.7655797004699707]], [[-0.5314797163009644]], [[1.6241461038589478]], [[3.023499011993408]], [[-2.1023881435394287]], [[-0.2512974739074707]], [[-0.5788166522979736]], [[0.5905413627624512]], [[4.765209197998047]], [[0.3375716209411621]], [[1.6312397718429565]], [[-1.0770684480667114]], [[-2.890617847442627]], [[-0.3959663510322571]], [[2.970667600631714]], [[0.4911074936389923]], [[-0.4825342297554016]], [[4.155728340148926]], [[4.661104679107666]], [[-0.1898704171180725]], [[2.1586761474609375]], [[2.1730074882507324]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a44b3d099d891cb83a0e3ab174130a74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a06c6af1188ca33554585de4e524cb44
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.5735554695129395]], [[-2.517559051513672]], [[1.0754027366638184]], [[-1.106138825416565]], [[-0.6882517337799072]], [[-1.5263524055480957]], [[-0.6446751356124878]], [[-0.5816648006439209]], [[4.781009197235107]], [[-3.171358585357666]], [[-6.15999174118042]], [[-0.7728664875030518]], [[-0.5666549205780029]], [[-2.13090443611145]], [[2.755659818649292]], [[4.737791538238525]], [[2.576324224472046]], [[3.7361953258514404]], [[0.9948139786720276]], [[-1.899785041809082]], [[-5.950649261474609]], [[-0.15697109699249268]], [[0.14505717158317566]], [[4.234015464782715]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class PrimitiveOp_fc2e652b8f90d6d20700d6f7ebae0887(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4179, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[4179, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b23fb233d2c064fa6c8a5d4f0647d6ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc2e652b8f90d6d20700d6f7ebae0887
    def get_inputs(self):
        return [
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b23fb233d2c064fa6c8a5d4f0647d6ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc2e652b8f90d6d20700d6f7ebae0887
    def get_inputs(self):
        return [
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b23fb233d2c064fa6c8a5d4f0647d6ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc2e652b8f90d6d20700d6f7ebae0887
    def get_inputs(self):
        return [
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b23fb233d2c064fa6c8a5d4f0647d6ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc2e652b8f90d6d20700d6f7ebae0887
    def get_inputs(self):
        return [
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b23fb233d2c064fa6c8a5d4f0647d6ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc2e652b8f90d6d20700d6f7ebae0887
    def get_inputs(self):
        return [
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b23fb233d2c064fa6c8a5d4f0647d6ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc2e652b8f90d6d20700d6f7ebae0887
    def get_inputs(self):
        return [
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
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


class PrimitiveOp_a1248105acb78ae946a2c68cb5356c31(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4662, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[4662, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_69546f1fb3fcbb7849102dafedefa9d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1248105acb78ae946a2c68cb5356c31
    def get_inputs(self):
        return [
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_69546f1fb3fcbb7849102dafedefa9d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1248105acb78ae946a2c68cb5356c31
    def get_inputs(self):
        return [
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_69546f1fb3fcbb7849102dafedefa9d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1248105acb78ae946a2c68cb5356c31
    def get_inputs(self):
        return [
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_69546f1fb3fcbb7849102dafedefa9d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1248105acb78ae946a2c68cb5356c31
    def get_inputs(self):
        return [
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_69546f1fb3fcbb7849102dafedefa9d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1248105acb78ae946a2c68cb5356c31
    def get_inputs(self):
        return [
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_69546f1fb3fcbb7849102dafedefa9d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1248105acb78ae946a2c68cb5356c31
    def get_inputs(self):
        return [
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
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


class PrimitiveOp_31f711eb9a78b73191b19c7e06f3c6a7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3857, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[3857, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ce148ef8b57b53f06895a5418bab5a37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31f711eb9a78b73191b19c7e06f3c6a7
    def get_inputs(self):
        return [
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ce148ef8b57b53f06895a5418bab5a37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31f711eb9a78b73191b19c7e06f3c6a7
    def get_inputs(self):
        return [
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ce148ef8b57b53f06895a5418bab5a37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31f711eb9a78b73191b19c7e06f3c6a7
    def get_inputs(self):
        return [
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ce148ef8b57b53f06895a5418bab5a37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31f711eb9a78b73191b19c7e06f3c6a7
    def get_inputs(self):
        return [
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ce148ef8b57b53f06895a5418bab5a37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31f711eb9a78b73191b19c7e06f3c6a7
    def get_inputs(self):
        return [
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ce148ef8b57b53f06895a5418bab5a37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31f711eb9a78b73191b19c7e06f3c6a7
    def get_inputs(self):
        return [
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
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