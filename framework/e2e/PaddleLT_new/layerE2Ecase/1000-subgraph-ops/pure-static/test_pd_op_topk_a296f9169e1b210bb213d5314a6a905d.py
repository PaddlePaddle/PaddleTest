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



class PrimitiveOp_5288a1eb3b86298105828e5c9ba6f5a7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.topk(input_0, input_1, 2, True, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 17, 50, 76], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fafb398d45ace9c8ff8f91ce1fa63d54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5288a1eb3b86298105828e5c9ba6f5a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([4], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_6a6a207ba3c0cc77e7d0970313bd33b5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.topk(input_0, input_1, 2, True, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 17, 100, 152], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_38b6980746926b38c4826e9f20a45145(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a6a207ba3c0cc77e7d0970313bd33b5
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([4], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_baf4a88a74595c6ae9d3006e20b692ac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.topk(input_0, input_1, 2, True, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 17, 7, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_45b931670ec23345c62fbbc45f4ee312(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_baf4a88a74595c6ae9d3006e20b692ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 7, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([4], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_00ae5c554e24f47f11ef53a82fa6875e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.topk(input_0, input_1, 2, True, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 17, 13, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_862e142f95d72a78b90dc0642556bd6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00ae5c554e24f47f11ef53a82fa6875e
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 13, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([4], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_20475a8cdee78e05d847803d7cfcc010(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.topk(input_0, input_1, 2, True, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 17, 25, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2c7fe3bb576c00f69d5cc775c6a04a4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20475a8cdee78e05d847803d7cfcc010
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([4], dtype='int32').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()