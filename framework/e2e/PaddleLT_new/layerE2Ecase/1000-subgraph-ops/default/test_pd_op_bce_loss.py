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



class PrimitiveOp_3b830507f98b4d952324f61fa64fa9bc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.bce_loss(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 81], dtype='float32'),
            paddle.static.InputSpec(shape=[3800, 81], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a44bbf4609726c9066f1040e8afe72e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b830507f98b4d952324f61fa64fa9bc
    def get_inputs(self):
        return [
            paddle.uniform([3800, 81], dtype='float32', min=0, max=0.5),
            paddle.uniform([3800, 81], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c77ff64c0a010b26e790c7c24f0045f5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.bce_loss(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f884ef2d2055106659d901ea89d854f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c77ff64c0a010b26e790c7c24f0045f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_dca694a1b2818a2c6b93ba1658ddf63f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.bce_loss(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 81], dtype='float32'),
            paddle.static.InputSpec(shape=[15200, 81], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7e752fd6eec45e0e3879dd4b0121dad2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dca694a1b2818a2c6b93ba1658ddf63f
    def get_inputs(self):
        return [
            paddle.uniform([15200, 81], dtype='float32', min=0, max=0.5),
            paddle.uniform([15200, 81], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_60cfefb3806c988a13e7abd11053d405(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.bce_loss(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 81], dtype='float32'),
            paddle.static.InputSpec(shape=[70, 81], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f11594a94bf86f86f2e1dc8f0f5bf9f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60cfefb3806c988a13e7abd11053d405
    def get_inputs(self):
        return [
            paddle.uniform([70, 81], dtype='float32', min=0, max=0.5),
            paddle.uniform([70, 81], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5b205d359d7eb2cb2f55344a0a1de9e4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.bce_loss(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 81], dtype='float32'),
            paddle.static.InputSpec(shape=[247, 81], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7d1cdbb990f5281fc42e781d86c108c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b205d359d7eb2cb2f55344a0a1de9e4
    def get_inputs(self):
        return [
            paddle.uniform([247, 81], dtype='float32', min=0, max=0.5),
            paddle.uniform([247, 81], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d96e4dfae9bbaa89001ca751db8583ea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.bce_loss(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 81], dtype='float32'),
            paddle.static.InputSpec(shape=[950, 81], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ff9425999b4b216a09a6ee9a99412760(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d96e4dfae9bbaa89001ca751db8583ea
    def get_inputs(self):
        return [
            paddle.uniform([950, 81], dtype='float32', min=0, max=0.5),
            paddle.uniform([950, 81], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()