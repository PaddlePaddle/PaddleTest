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



class PrimitiveOp_e419f4b51dcd1683fa62a710a7e48322(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1, 1]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 300, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8ca5a97ae9bb1b8290b932ab5796826c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e419f4b51dcd1683fa62a710a7e48322
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1, 1], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_39264d88a0913daf252286a97f3f9573(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1, 4]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7d63e817a6041f656af4ffb964748ece(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39264d88a0913daf252286a97f3f9573
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_42c392c241c0aa71a45c9c86076d6e02(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1, 68]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3dd7982b0204807335be3b6dc97e4009(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42c392c241c0aa71a45c9c86076d6e02
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_8e473e34f4608ad36878beda070269a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39264d88a0913daf252286a97f3f9573
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 11109, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_78261f5374d99401f4a96dec21f0c57d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42c392c241c0aa71a45c9c86076d6e02
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 11109, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_7d63e817a6041f656af4ffb964748ece(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39264d88a0913daf252286a97f3f9573
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_2af60f87d9de8632daf3b50b9282e5dc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1, 76]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bff53b5afaa0f14ec1125247b3c57911(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2af60f87d9de8632daf3b50b9282e5dc
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 76], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_ef4b41a513ac4c5adc3ad73d0805360f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39264d88a0913daf252286a97f3f9573
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3024, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_6fadecd46817346ed472497597644ccf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42c392c241c0aa71a45c9c86076d6e02
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3024, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_d86a83559666a57df02eb23bf3305faf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39264d88a0913daf252286a97f3f9573
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_17b6e7afc7f5d4d539b981376160f8d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42c392c241c0aa71a45c9c86076d6e02
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_9218528e3d20c4002d6ac7cccf471001(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39264d88a0913daf252286a97f3f9573
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 9261, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_ca283989ca94271fd5ea0c74fb79fb12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42c392c241c0aa71a45c9c86076d6e02
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 9261, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_c5a2c0eefa21bba13b301491c1d299dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39264d88a0913daf252286a97f3f9573
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 2100, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_55f13f89f4945ae64680449d14c2fb13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42c392c241c0aa71a45c9c86076d6e02
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 2100, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_6b978e6121628346940655ee683b1e91(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 100, 1]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, None], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_602b2c4ee3dea54019650f224b5cde6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b978e6121628346940655ee683b1e91
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.42495518922805786, 0.2379216104745865, 0.12352313846349716, 0.09334713220596313]]], dtype='float32').reshape([1, 1, 4]),
            paddle.to_tensor([1, 100, 1], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_c2e90eaab9a75629f3308df3d581be84(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 300, 1]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, None], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b191b2f98a86aaf26f312289a33e8430(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2e90eaab9a75629f3308df3d581be84
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.4153171479701996, 0.3734067678451538, 0.3560805022716522, 0.21529623866081238]]], dtype='float32').reshape([1, 1, 4]),
            paddle.to_tensor([1, 300, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_3fc8292c981b5c49e2e5f17439030a1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39264d88a0913daf252286a97f3f9573
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4725, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_a10b64187276a93a65e54c0d8ef284fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42c392c241c0aa71a45c9c86076d6e02
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4725, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_b6ac6ca8502b3b4602e3e0d38a4c5519(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39264d88a0913daf252286a97f3f9573
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 6069, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_e4a23d920073eb13626a776652820ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42c392c241c0aa71a45c9c86076d6e02
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 6069, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_c033661296d2211ebe1cf8040abdd330(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39264d88a0913daf252286a97f3f9573
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 7581, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_6e5bed5084c3df0692debec08e6945f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42c392c241c0aa71a45c9c86076d6e02
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 7581, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_8b57bf9879aa91ed64d4eeb65398692e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1, 512]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b228c53b81614eb87f5787e3240a71c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b57bf9879aa91ed64d4eeb65398692e
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1, 512], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_d86a83559666a57df02eb23bf3305faf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39264d88a0913daf252286a97f3f9573
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_17b6e7afc7f5d4d539b981376160f8d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42c392c241c0aa71a45c9c86076d6e02
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_b228c53b81614eb87f5787e3240a71c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b57bf9879aa91ed64d4eeb65398692e
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1, 512], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_bc5e8cdd7ef1d0fd9d1664d7f1afcae7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39264d88a0913daf252286a97f3f9573
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 8400, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_ae44d491d81b31b1ea06f00ae6a13ce2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42c392c241c0aa71a45c9c86076d6e02
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 8400, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]




if __name__ == '__main__':
    unittest.main()