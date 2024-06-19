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



class PrimitiveOp_80ab0885dd242442f92a7d720e75d3bc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1c572d00896bd98ac95805b2faf6b36a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80ab0885dd242442f92a7d720e75d3bc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.4572192430496216, -0.2408244013786316, -0.0878177285194397, 0.1381824016571045]]], dtype='float32').reshape([1, 1, 4]),
            paddle.to_tensor([1, 100, 1], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c3296805b640241c669460f39dd215aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 3024, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_cddc99d669bbad8da6dfca7e2bf0afbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 3024, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_04a9f3b4ac91a9104add8f74d0a67319(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 4725, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_131528afdfb29214f9bc0348048c2b6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 4725, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_e267c561dd99752cf80ade80e3be542d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80ab0885dd242442f92a7d720e75d3bc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.2676364779472351, -0.12877234816551208, -0.41700518131256104, -0.3967134952545166]]], dtype='float32').reshape([1, 1, 4]),
            paddle.to_tensor([1, 300, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_49eafc2ff2c0ae2286a17e9721d07bba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80ab0885dd242442f92a7d720e75d3bc
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1, 512], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_9e85e00b4d5b927cddc7af4e331db445(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80ab0885dd242442f92a7d720e75d3bc
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_73722ba711f78b2302a254e9e3992d0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 4116, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_01343d8fb616db47a74a54756c8a73fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 4116, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_5e3b75fc6d556e79a8ceed1463178f9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 3549, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_71b54512a88bb7b6c16e04b08c9fce38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 3549, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 76], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_a456f977e2602d38864297a9b316c423(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 6069, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_e7ceabe6aae814c78e68a53189e71c9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 6069, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_73722ba711f78b2302a254e9e3992d0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 4116, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_01343d8fb616db47a74a54756c8a73fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 4116, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_7103509cc06b3a389f3d6cdc62e97632(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 11109, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_8fd9d06d7ba71da6f4ea0ec1f38765f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 11109, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_b5e6ba02770ac41807743955ef1e8163(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 2100, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_c4c6a912327a70a48e5a7c079697a01a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 2100, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_5e3b75fc6d556e79a8ceed1463178f9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 3549, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_ef9eeea3d60b5ec43cab28c7dcef1ef3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 3549, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_49eafc2ff2c0ae2286a17e9721d07bba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80ab0885dd242442f92a7d720e75d3bc
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1, 512], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_f4654da4e5fa127a3f24c4cf3e4868d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 8400, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_b9d3bf9d9163c78a2ac058f4f3d59344(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 8400, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_8493bfda5ed12c77759c36c3e4a5762b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 9261, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_8825b135759cb5d325082bf542194f0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 9261, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_80f043f9a655842e93ef3738c63dfe5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 7581, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_b6069fd1be901bf323ce7fd0b76175b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cb55b7e5a0a04a28b9e3a520deb88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 7581, 1], dtype='int32'),
            paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
        ]




if __name__ == '__main__':
    unittest.main()