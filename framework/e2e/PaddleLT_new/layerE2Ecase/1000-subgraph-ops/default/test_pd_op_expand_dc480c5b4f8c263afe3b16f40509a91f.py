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



class PrimitiveOp_86fc2db2c5c00a46b77847df1040e142(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1, 24, 36]
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7136619b30ec9fc4dbc68102a8736fb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86fc2db2c5c00a46b77847df1040e142
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1, 24, 36], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_7136619b30ec9fc4dbc68102a8736fb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86fc2db2c5c00a46b77847df1040e142
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


class PrimitiveOp_a7d9f435df87dcec37c8f4e42cab7eb3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 256, 21]
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dbf86d08162be483205b73276bc20bf0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7d9f435df87dcec37c8f4e42cab7eb3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 21], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 256, 21], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_165aa28d4dce34292e9678f866483359(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1, 25, 38]
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7a9b6fd1095ad55526d52a6b0b0ea03e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_165aa28d4dce34292e9678f866483359
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1, 25, 38], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_7a9b6fd1095ad55526d52a6b0b0ea03e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_165aa28d4dce34292e9678f866483359
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1, 25, 38], dtype='int64').reshape([4]),
        ]


class PrimitiveOp_55bea5cd3728849ceb3a6db7c1fd8e55(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1, 20, 30]
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eaeb8523d674caf0dad7c22399992ecd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55bea5cd3728849ceb3a6db7c1fd8e55
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1, 20, 30], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eaeb8523d674caf0dad7c22399992ecd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55bea5cd3728849ceb3a6db7c1fd8e55
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


class PrimitiveOp_16d8f022f1e226d1adb5ad3c42a7b764(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [7, 256, 19]
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2f4a38af5878cf5b8399b59f6d3e48e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16d8f022f1e226d1adb5ad3c42a7b764
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([7, 256, 19], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_2dbc23559908a20b8be6c30c227f3289(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1, 15, 25]
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7af8052de5e3a1f917ff75ef8cc46680(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2dbc23559908a20b8be6c30c227f3289
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 15, 25], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1, 15, 25], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_7af8052de5e3a1f917ff75ef8cc46680(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2dbc23559908a20b8be6c30c227f3289
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


class PrimitiveOp_31c1667866879b3484ea3d4b6b8c0ca7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [8, 256, 150]
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b5ee5747a53e588969aaac91f5cfdc34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31c1667866879b3484ea3d4b6b8c0ca7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 150], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8, 256, 150], dtype='int64').reshape([3]),
        ]




if __name__ == '__main__':
    unittest.main()