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
        return False
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



class PrimitiveOp_e3bff0c9b4c2a62367b317a8e2d7b430(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.one_hot(input_0 % input_1, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d8ac18a7c9b7c8c17e3029d7b9985002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3bff0c9b4c2a62367b317a8e2d7b430
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 4, 1, 5], dtype='int32').reshape([4]),
            paddle.to_tensor([80], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_0fc23c972264c665ebf6d6bb1bb0687f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.one_hot(input_0 % input_1, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f1a37ff251d37210897237d60b98348f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0fc23c972264c665ebf6d6bb1bb0687f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
            paddle.to_tensor([21], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_33fcee62331fa60f21e65e54a9a4b117(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3bff0c9b4c2a62367b317a8e2d7b430
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 0, 3], dtype='int32').reshape([3]),
            paddle.to_tensor([80], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_3034091ef0340bcd775eafb8cb5671f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0fc23c972264c665ebf6d6bb1bb0687f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
            paddle.to_tensor([81], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_28bc5191f97e9092d1ce8b69f26904f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0fc23c972264c665ebf6d6bb1bb0687f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
            paddle.to_tensor([21], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_3b5b16045b52eeda21625773bc7dfae9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3bff0c9b4c2a62367b317a8e2d7b430
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3, 5, 3, 5, 6], dtype='int32').reshape([6]),
            paddle.to_tensor([80], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_f838ea7c051e868c26ccbb1550789ce4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3bff0c9b4c2a62367b317a8e2d7b430
    def get_inputs(self):
        return [
            paddle.to_tensor([9, 1], dtype='int32').reshape([2]),
            paddle.to_tensor([80], dtype='int32').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()