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



class PrimitiveOp_b92e5fad556ba52e1c8f1251eef629d4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-3, -3]
        return paddle._C_ops.roll(input_0, input_1, [1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 7, 7, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_17ba5df40f1c8243c9aa8db46ccdb652(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b92e5fad556ba52e1c8f1251eef629d4
    def get_inputs(self):
        return [
            paddle.uniform([43, 7, 7, 768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_88ad3aaa9b16bc3d76d3c600c74c9a37(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-3, -3]
        return paddle._C_ops.roll(input_0, input_1, [1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 14, 14, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_318d5e23fb18c254280107b634c48018(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88ad3aaa9b16bc3d76d3c600c74c9a37
    def get_inputs(self):
        return [
            paddle.uniform([43, 14, 14, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7fd2bb8be1fb4d6a8a1ca76175501266(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b92e5fad556ba52e1c8f1251eef629d4
    def get_inputs(self):
        return [
            paddle.uniform([11, 7, 7, 768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_4f6de8b90a640a0d87805dba4c921b5c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-3, -3]
        return paddle._C_ops.roll(input_0, input_1, [1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 56, 56, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8050b1b7ef01a646029bdeab7670f430(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f6de8b90a640a0d87805dba4c921b5c
    def get_inputs(self):
        return [
            paddle.uniform([43, 56, 56, 96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_6fcfeef6cbad776ce9e8c746e5de579b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-3, -3]
        return paddle._C_ops.roll(input_0, input_1, [1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 28, 28, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e77507e7fb9f976752481766ceb8a056(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fcfeef6cbad776ce9e8c746e5de579b
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 28, 192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_318d5e23fb18c254280107b634c48018(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88ad3aaa9b16bc3d76d3c600c74c9a37
    def get_inputs(self):
        return [
            paddle.uniform([43, 14, 14, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f9b712ddcbb7d1b6d1fb8a4edb4c67f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88ad3aaa9b16bc3d76d3c600c74c9a37
    def get_inputs(self):
        return [
            paddle.uniform([11, 14, 14, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_05fe8b0e7aff9b393fa605276f77d74d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f6de8b90a640a0d87805dba4c921b5c
    def get_inputs(self):
        return [
            paddle.uniform([11, 56, 56, 96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5a4d071585512364abb9cadcebed9b68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fcfeef6cbad776ce9e8c746e5de579b
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 28, 192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f9b712ddcbb7d1b6d1fb8a4edb4c67f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88ad3aaa9b16bc3d76d3c600c74c9a37
    def get_inputs(self):
        return [
            paddle.uniform([11, 14, 14, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
        ]




if __name__ == '__main__':
    unittest.main()