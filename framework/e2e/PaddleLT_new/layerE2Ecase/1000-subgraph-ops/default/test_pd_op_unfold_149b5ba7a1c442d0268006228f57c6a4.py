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


class TestPrimitiveOp_d5b1079b79d0cfccfbff9b141d8a5d42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5999375d1ea99c530aaf52bea9d8e468
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 112, 112], dtype='float32', min=0, max=0.5),
        ]


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


class TestPrimitiveOp_65fa64d8bfb1062495e144d160b500b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2109c311607c04cb3f34c23075d5526b
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_86da92f34b458cc82475ca4b5470fe81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2109c311607c04cb3f34c23075d5526b
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ec23cb6e1b0a6f550ce365883cee0dfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2109c311607c04cb3f34c23075d5526b
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_a34314445073bc2072b5685c8cd2e130(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d920e44141b5b0c5869a22e1f05352ee
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0fb5c40f31e4510e465fb893df87e622(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2109c311607c04cb3f34c23075d5526b
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_41e527c272fa6c012d84e8cfe82b1495(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d920e44141b5b0c5869a22e1f05352ee
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3c876007b6645dd848b9710020eb78a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2109c311607c04cb3f34c23075d5526b
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_12d6cc4d6dead4b12b3cf75028f9e7b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d920e44141b5b0c5869a22e1f05352ee
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e8dda9d1c89e1c04215e13d7e5ec80bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2109c311607c04cb3f34c23075d5526b
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b04153b13c56b22555fe56b5401441cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2109c311607c04cb3f34c23075d5526b
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ecd249837193d7bade6cac631055492c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d920e44141b5b0c5869a22e1f05352ee
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae0d865c88c95c721766f73a89700894(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d920e44141b5b0c5869a22e1f05352ee
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_59b509df7a0809121c3984c4b2a25110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5999375d1ea99c530aaf52bea9d8e468
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 112, 112], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ad1204fad4b77ad5efd89dc8eaaad315(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2109c311607c04cb3f34c23075d5526b
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_18e3a07956c60f972bdc0ec5876f8893(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d920e44141b5b0c5869a22e1f05352ee
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 28, 28], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()