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



class PrimitiveOp_bd5df49be6bf453ea4105ae4df0fd8f3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.unfold(input_0, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 64, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_759f662e37ea2c0e9fbe171dec98a7f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd5df49be6bf453ea4105ae4df0fd8f3
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 56, 56], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_46c299bb7cc6c525daa4e7019d0a2def(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.unfold(input_0, [7, 7], [2, 2], [3, 3, 3, 3], [1, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 512, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5aad794b45adcf5d2d5a994c57896cac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46c299bb7cc6c525daa4e7019d0a2def
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3bcf77301308305e69f2c78c48177800(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.unfold(input_0, [7, 7], [2, 2], [3, 3, 3, 3], [1, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 128, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_deef56fc7908c28e5b4d31013c45c5c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bcf77301308305e69f2c78c48177800
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 56, 56], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_de44c3602233099f5c9de9c24e28953c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.unfold(input_0, [3, 3], [1, 1], [1, 1, 1, 1], [1, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 32, 112, 112], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_61e7d02e7dff5dd97f20b3deffb35d6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de44c3602233099f5c9de9c24e28953c
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 112, 112], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ed12377adc37f84b899e7af67b60d0c3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.unfold(input_0, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 256, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5a45a6753c2b357999f24fc547e85fdd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed12377adc37f84b899e7af67b60d0c3
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d1c83866c2bca94fcc8a9b39c9a4de97(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.unfold(input_0, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 512, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4623329a6162ac3c7cb07c599bbc1f54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1c83866c2bca94fcc8a9b39c9a4de97
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_25f2a0ed8b99c44bb3bef3ea4b728b01(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.unfold(input_0, [7, 7], [2, 2], [3, 3, 3, 3], [1, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 128, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9b1f1b93b6d19b9929cc33348e57c64b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25f2a0ed8b99c44bb3bef3ea4b728b01
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 56, 56], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_38dba5fa0f4dbe290cca8f36ccf86490(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.unfold(input_0, [7, 7], [2, 2], [3, 3, 3, 3], [1, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 256, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_755a88ee2218f715f9cd71e9131a9bf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38dba5fa0f4dbe290cca8f36ccf86490
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 28, 28], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5a52bb67cb6866adfb7644f70dbd6dee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.unfold(input_0, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 128, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9e57b2d0c5d8f5a14f7618334b9ee84d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a52bb67cb6866adfb7644f70dbd6dee
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 28, 28], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_fa0f8034fd5f70b5766dcc9267c0129a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.unfold(input_0, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 128, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3616efed6ea1f954cd126c4e18e7f6cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa0f8034fd5f70b5766dcc9267c0129a
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 28, 28], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c1b258f6b4d74d8e1c8da22ea0c9de25(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.unfold(input_0, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 256, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_921b0eb4282f41f6a099cb504edb86c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b258f6b4d74d8e1c8da22ea0c9de25
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_179da825c28b3828b82f46c6b7f64a96(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.unfold(input_0, [7, 7], [2, 2], [3, 3, 3, 3], [1, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 512, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5c0839efa8ba958ef8b368f964baa66d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_179da825c28b3828b82f46c6b7f64a96
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d1b781e5313ffaf163aff7cd8e18e5b4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.unfold(input_0, [3, 3], [1, 1], [1, 1, 1, 1], [1, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 32, 112, 112], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bbb9ef92f0c5b88d53be272197140038(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1b781e5313ffaf163aff7cd8e18e5b4
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 112, 112], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_30b9daf5fafbbc544bd98a3aeb750765(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.unfold(input_0, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 512, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_733169864d0d1289c73d558606fbac3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30b9daf5fafbbc544bd98a3aeb750765
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_31ef6a87cc6895f8d13a2abd4fb5dac0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.unfold(input_0, [7, 7], [2, 2], [3, 3, 3, 3], [1, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 256, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a2729ef81d0d8bfc3fb1efc6c5ad2288(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31ef6a87cc6895f8d13a2abd4fb5dac0
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 28, 28], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6daac09b6cb78993a2ebe38ef5df73f0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.unfold(input_0, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 64, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c8da6c73e331e83221cb23bb0cc30be3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6daac09b6cb78993a2ebe38ef5df73f0
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 56, 56], dtype='float32', min=-0.5, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()