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



class PrimitiveOp_53556ad456cb1829e7df5ef4d58319fa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-3, -3]
        return paddle._C_ops.roll(input_0, input_1, [1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 7, 7, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_864f5948b9e6afed11f592a98b29ec3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53556ad456cb1829e7df5ef4d58319fa
    def get_inputs(self):
        return [
            paddle.uniform([43, 7, 7, 768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_00f747e0d3c94ba81d80a45f1a8f98f0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-3, -3]
        return paddle._C_ops.roll(input_0, input_1, [1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 14, 14, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6de85869a313160408d8bbe62db53c2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00f747e0d3c94ba81d80a45f1a8f98f0
    def get_inputs(self):
        return [
            paddle.uniform([43, 14, 14, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_5fcab53fba26ff4da47e7d5744605beb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-3, -3]
        return paddle._C_ops.roll(input_0, input_1, [1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 7, 7, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0177da730cfd85741d6a708b0d212c51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fcab53fba26ff4da47e7d5744605beb
    def get_inputs(self):
        return [
            paddle.uniform([11, 7, 7, 768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_9b7bf50b2fcea4b0a6dccecd621c7902(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-3, -3]
        return paddle._C_ops.roll(input_0, input_1, [1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 56, 56, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2643f67b4e88ad2a5ee9df87cbdabd86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b7bf50b2fcea4b0a6dccecd621c7902
    def get_inputs(self):
        return [
            paddle.uniform([43, 56, 56, 96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_a6024508c23564cb1a62d8250f8c8520(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-3, -3]
        return paddle._C_ops.roll(input_0, input_1, [1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 28, 28, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_22bd69c6d78efccdfc88b16a4c01b18c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6024508c23564cb1a62d8250f8c8520
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 28, 192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6de85869a313160408d8bbe62db53c2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00f747e0d3c94ba81d80a45f1a8f98f0
    def get_inputs(self):
        return [
            paddle.uniform([43, 14, 14, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_982bc5292fe08a4c562f1b4106a1552b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-3, -3]
        return paddle._C_ops.roll(input_0, input_1, [1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 14, 14, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ec901ad8e9cc6516772adf58f052f8c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_982bc5292fe08a4c562f1b4106a1552b
    def get_inputs(self):
        return [
            paddle.uniform([11, 14, 14, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_7973a7019a2166706022ce95e01f4ece(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-3, -3]
        return paddle._C_ops.roll(input_0, input_1, [1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 56, 56, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dff297d4473ac5e1df89213304a1b073(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7973a7019a2166706022ce95e01f4ece
    def get_inputs(self):
        return [
            paddle.uniform([11, 56, 56, 96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_ac43c60a843c7cef6150293e0b41dcba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-3, -3]
        return paddle._C_ops.roll(input_0, input_1, [1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 28, 28, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_de0e8354906093bdcb67e27216b1c4f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac43c60a843c7cef6150293e0b41dcba
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 28, 192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ec901ad8e9cc6516772adf58f052f8c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_982bc5292fe08a4c562f1b4106a1552b
    def get_inputs(self):
        return [
            paddle.uniform([11, 14, 14, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
        ]




if __name__ == '__main__':
    unittest.main()