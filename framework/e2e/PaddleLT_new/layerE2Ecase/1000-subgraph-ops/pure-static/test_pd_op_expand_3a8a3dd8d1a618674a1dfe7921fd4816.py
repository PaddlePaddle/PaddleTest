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



class PrimitiveOp_8c86a7d18dc7a4b3cb3db4e9c3d11def(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1, 24, 36]
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 24, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8527394e62da0b21ed23865daf3518ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c86a7d18dc7a4b3cb3db4e9c3d11def
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1, 24, 36], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_8527394e62da0b21ed23865daf3518ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c86a7d18dc7a4b3cb3db4e9c3d11def
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1, 24, 36], dtype='int64').reshape([4]),
        ]


class PrimitiveOp_ff6090f1bdaf41fe8d0f0ba4f75bae45(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, -1, -1]
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 150, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dde9161825a86f62db38142f35c24376(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff6090f1bdaf41fe8d0f0ba4f75bae45
    def get_inputs(self):
        return [
            paddle.uniform([1, 150, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, -1], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_669a242f86bce7a4cfd0beffe08afd6d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 256, 21]
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 21], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dae33a0fb12fe1dfc8133959d58f1d1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_669a242f86bce7a4cfd0beffe08afd6d
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 21], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 256, 21], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_01a5d3d5ff4c1baf28fddd0fa689bbb3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1, 25, 38]
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 25, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d22413c2c2757db52e7f5d52511c33f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01a5d3d5ff4c1baf28fddd0fa689bbb3
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1, 25, 38], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_d22413c2c2757db52e7f5d52511c33f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01a5d3d5ff4c1baf28fddd0fa689bbb3
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1, 25, 38], dtype='int64').reshape([4]),
        ]


class PrimitiveOp_beab76917c4b0d9b7f9b33a48812eaed(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1, 20, 30]
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 20, 30], dtype='float32'),
            paddle.static.InputSpec(shape=[4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2c67d4843e65dedb8ae6edefcb84612e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beab76917c4b0d9b7f9b33a48812eaed
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1, 20, 30], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_2c67d4843e65dedb8ae6edefcb84612e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beab76917c4b0d9b7f9b33a48812eaed
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1, 20, 30], dtype='int64').reshape([4]),
        ]


class PrimitiveOp_c93c1435fe299ddb22ff4a886ed4d84d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, -1, -1]
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7271b19204ba5faa3e5561734cd6d72a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c93c1435fe299ddb22ff4a886ed4d84d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, -1], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_b6ccaea197f54ed1d38f1b21ef701338(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [7, 256, 19]
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9cdce1851f9ede30a711619b22740016(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6ccaea197f54ed1d38f1b21ef701338
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([7, 256, 19], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_4de2581c9b48b1671a93858290f310f4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1, 15, 25]
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 15, 25], dtype='float32'),
            paddle.static.InputSpec(shape=[4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_767721cea1fb3a7ce09c8dc0b31cceb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4de2581c9b48b1671a93858290f310f4
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 15, 25], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1, 15, 25], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_767721cea1fb3a7ce09c8dc0b31cceb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4de2581c9b48b1671a93858290f310f4
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 15, 25], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1, 15, 25], dtype='int64').reshape([4]),
        ]


class PrimitiveOp_314e9840d096ba747eb1891a130c981b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, -1, -1]
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 150, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1d1952fdc0738f306ebebddb29aad085(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_314e9840d096ba747eb1891a130c981b
    def get_inputs(self):
        return [
            paddle.uniform([1, 150, 768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, -1], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_4ba3dfb5e3e732223b89125f0a073a15(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [8, 256, 150]
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 150], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9927dcc76fd1bcccdf567808cc65b1fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ba3dfb5e3e732223b89125f0a073a15
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 150], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8, 256, 150], dtype='int64').reshape([3]),
        ]




if __name__ == '__main__':
    unittest.main()