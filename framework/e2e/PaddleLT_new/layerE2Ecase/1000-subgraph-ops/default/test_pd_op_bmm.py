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



class PrimitiveOp_5766c971abeb3c3cc456767f4b484825(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.bmm(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 19, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7a483bff11b2890acec4a27a5e770ed6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5766c971abeb3c3cc456767f4b484825
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 32768], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32768, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7975384f4b3f3bc6976b396512594b77(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.bmm(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 21, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2d7674035d0e50d96b86b9a765114b63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7975384f4b3f3bc6976b396512594b77
    def get_inputs(self):
        return [
            paddle.uniform([1, 21, 16384], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 16384, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3844321aca602c1c96b9febb59925918(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.bmm(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 64, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f68902290806ed60b30f9b89d40daba6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3844321aca602c1c96b9febb59925918
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 4096], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9ff31d6105f89182d14461fa4ceadb1c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.bmm(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a75d0343c38915c898be7b481135fd5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ff31d6105f89182d14461fa4ceadb1c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 4096], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4096, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9d7e374ac35f30772f71913c4edc7211(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3844321aca602c1c96b9febb59925918
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 8192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c605765df13a7e8c042f37dd3697bc9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ff31d6105f89182d14461fa4ceadb1c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 8192], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8192, 8192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_088aea3b1f93742feaa5e556501788e0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.bmm(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4b2b41db4f4683c71aedb8d7acfc6075(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_088aea3b1f93742feaa5e556501788e0
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 8192], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8192, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cc94d97785579c630c39a52e52a4b8ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.bmm(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 512, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d0581e998b735e0fd03c5d1731e496d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc94d97785579c630c39a52e52a4b8ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 8192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1f53677e448dc14950448a4e790f43d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_088aea3b1f93742feaa5e556501788e0
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 4096], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4096, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9799a117cc43bf011f5930ca23c89152(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc94d97785579c630c39a52e52a4b8ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 4096], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()