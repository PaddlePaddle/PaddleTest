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



class PrimitiveOp_2109c311607c04cb3f34c23075d5526b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.unfold(input_0, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_940c55ca40ba3deb2012c542e838e538(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2109c311607c04cb3f34c23075d5526b
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 56, 56], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d920e44141b5b0c5869a22e1f05352ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.unfold(input_0, [7, 7], [2, 2], [3, 3, 3, 3], [1, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_49074e807f3ecdde82b7c57a5e2c84e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d920e44141b5b0c5869a22e1f05352ee
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8f3eea4857ea3cfb30f17dbd3cb9144c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d920e44141b5b0c5869a22e1f05352ee
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 56, 56], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5999375d1ea99c530aaf52bea9d8e468(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.unfold(input_0, [3, 3], [1, 1], [1, 1, 1, 1], [1, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a64cb7e62c367c8f91ad0c06644fd568(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5999375d1ea99c530aaf52bea9d8e468
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 112, 112], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b34101f28ee1bcfb1ff9723787d484b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2109c311607c04cb3f34c23075d5526b
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c9c0d81d07980ed721dfe2e05346f65f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2109c311607c04cb3f34c23075d5526b
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_38fdd8979680417efc35fa189ff304a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d920e44141b5b0c5869a22e1f05352ee
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 56, 56], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_951729ede87282e79176dbbb781041d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d920e44141b5b0c5869a22e1f05352ee
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 28, 28], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e723075f602e6c0e794948fa3dbf3dd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2109c311607c04cb3f34c23075d5526b
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 28, 28], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_da00f1308db94ce1bdf2e4f57c52edbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2109c311607c04cb3f34c23075d5526b
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 28, 28], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b5c07b000a89e2bc9d5a5aa8bba0b43a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2109c311607c04cb3f34c23075d5526b
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_39bba3014cad1efda5242631a581ef84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d920e44141b5b0c5869a22e1f05352ee
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d52738e86b2aaf4afc29625357920b44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5999375d1ea99c530aaf52bea9d8e468
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 112, 112], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e30820775a1c44ffd658961e4b481391(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2109c311607c04cb3f34c23075d5526b
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_81f7a5b002c301075596447f4284e0cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d920e44141b5b0c5869a22e1f05352ee
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 28, 28], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2bbf1fa7615359233979edc3083b14bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2109c311607c04cb3f34c23075d5526b
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 56, 56], dtype='float32', min=-0.5, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()