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



class PrimitiveOp_ea8ea6ed689c6462a1d25c8810bf1cdf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.clip(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 72], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4cc7dbe81312848274a63a8a7e213f56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea8ea6ed689c6462a1d25c8810bf1cdf
    def get_inputs(self):
        return [
            paddle.uniform([1, 72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_2e8a1da01b58e325bad9e6ef1edb4405(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.clip(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_305cdcb1fa860e69e433b0a187f0b7b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2e8a1da01b58e325bad9e6ef1edb4405
    def get_inputs(self):
        return [
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c9571f1545ebeeef7f43f5ea9dc793f2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.clip(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 960], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2e03e5bd3bc8fef24e97cf86d01f5f04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9571f1545ebeeef7f43f5ea9dc793f2
    def get_inputs(self):
        return [
            paddle.uniform([1, 960], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_101546c092c13f4aeef5032773a9ee5c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.clip(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 480], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d2229ba526020af1d65822d3b34ffaa1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_101546c092c13f4aeef5032773a9ee5c
    def get_inputs(self):
        return [
            paddle.uniform([1, 480], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_cc7c3329af58c980f91d11faf6d58c9f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.clip(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_406ff178ddc6662f3f4b9dcdfe326ac5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc7c3329af58c980f91d11faf6d58c9f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.1611597239971161]], [[0.49395954608917236]], [[0.2646945118904114]], [[0.11859605461359024]], [[0.11985684931278229]], [[0.11257714033126831]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([-10000000000.0], dtype='float32').reshape([1]),
            paddle.to_tensor([4.135169982910156], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.clip(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_10861d25202c9c1b2ea0f4763f829bf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_10861d25202c9c1b2ea0f4763f829bf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_10861d25202c9c1b2ea0f4763f829bf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_10861d25202c9c1b2ea0f4763f829bf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_a757c2d16b1a5298599ec6a75f8b1ec2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.clip(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 336], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_46de2d8c4d297c229a90aa852e6d6b5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a757c2d16b1a5298599ec6a75f8b1ec2
    def get_inputs(self):
        return [
            paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2f0eac75f403b55095317dc88c1c2480(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2f0eac75f403b55095317dc88c1c2480(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2f0eac75f403b55095317dc88c1c2480(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2f0eac75f403b55095317dc88c1c2480(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_be491969ce50968f5ce95154e0c848d7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.clip(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_afa61897ae5b12ad89857c71f339ad1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be491969ce50968f5ce95154e0c848d7
    def get_inputs(self):
        return [
            paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_076d2d4765ad9eb281271760246cabe6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_076d2d4765ad9eb281271760246cabe6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_076d2d4765ad9eb281271760246cabe6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_076d2d4765ad9eb281271760246cabe6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8f51e5ab87412f794f78a6adbdb26196(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8f51e5ab87412f794f78a6adbdb26196(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8f51e5ab87412f794f78a6adbdb26196(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8f51e5ab87412f794f78a6adbdb26196(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_abda45cd897c6a197a0c6be62515e010(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.clip(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_32ffe7fce3e6a3062e2e2d344eaa8911(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_abda45cd897c6a197a0c6be62515e010
    def get_inputs(self):
        return [
            paddle.to_tensor(8732.0, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_71ecc94f3a74e2b415e74e24f31d4499(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a757c2d16b1a5298599ec6a75f8b1ec2
    def get_inputs(self):
        return [
            paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_71ecc94f3a74e2b415e74e24f31d4499(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a757c2d16b1a5298599ec6a75f8b1ec2
    def get_inputs(self):
        return [
            paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cad6dbd7929691bd68b778d716d4580c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cad6dbd7929691bd68b778d716d4580c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cad6dbd7929691bd68b778d716d4580c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cad6dbd7929691bd68b778d716d4580c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_30d54fd556c75f31938966f90b72b6ba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.clip(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_31c7243150e7e166529428d3cdfa6321(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30d54fd556c75f31938966f90b72b6ba
    def get_inputs(self):
        return [
            paddle.uniform([145, 240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2fba229405fef0b206732276954abb97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2fba229405fef0b206732276954abb97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2fba229405fef0b206732276954abb97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2fba229405fef0b206732276954abb97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_10861d25202c9c1b2ea0f4763f829bf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_10861d25202c9c1b2ea0f4763f829bf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_10861d25202c9c1b2ea0f4763f829bf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_10861d25202c9c1b2ea0f4763f829bf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2a59c2426f279e05c8e4901a7347b86c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2a59c2426f279e05c8e4901a7347b86c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2a59c2426f279e05c8e4901a7347b86c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2a59c2426f279e05c8e4901a7347b86c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_580671641ddab915889a07c4091a4fe6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be491969ce50968f5ce95154e0c848d7
    def get_inputs(self):
        return [
            paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_309abcf95474782b38aa9a4ee326ecde(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.clip(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 872], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e129881a4284943e8a925d51fbb8a529(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_309abcf95474782b38aa9a4ee326ecde
    def get_inputs(self):
        return [
            paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_35e0fa918b72f3c579941e75db76dcbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_35e0fa918b72f3c579941e75db76dcbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_35e0fa918b72f3c579941e75db76dcbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_35e0fa918b72f3c579941e75db76dcbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.clip(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bb698dce38985cb5bb4dbb06890a07bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bb698dce38985cb5bb4dbb06890a07bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e0e229d1c11ddc8ff67de2ce916a398b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc7c3329af58c980f91d11faf6d58c9f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_64d0a0a8ece307d4498d0f6e58926e89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_64d0a0a8ece307d4498d0f6e58926e89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_64d0a0a8ece307d4498d0f6e58926e89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_64d0a0a8ece307d4498d0f6e58926e89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_97264156dc95c9322abc0e26c08db3e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_97264156dc95c9322abc0e26c08db3e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_97264156dc95c9322abc0e26c08db3e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_97264156dc95c9322abc0e26c08db3e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_387e6fe476d87d4b93ab486b92156895(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a757c2d16b1a5298599ec6a75f8b1ec2
    def get_inputs(self):
        return [
            paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2a3d79a58e255115a369aa053a54c61d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2a3d79a58e255115a369aa053a54c61d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2a3d79a58e255115a369aa053a54c61d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2a3d79a58e255115a369aa053a54c61d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_562b4c45c53016e05942b67a6ab8018e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_562b4c45c53016e05942b67a6ab8018e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_562b4c45c53016e05942b67a6ab8018e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_562b4c45c53016e05942b67a6ab8018e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_f3bd307b0a33d3c8eb57a86633fe32cc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.clip(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9cab6dc8f3cd6b6a584d407e7fefbc61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3bd307b0a33d3c8eb57a86633fe32cc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.007246673107147217], [-0.43975457549095154], [-0.40652668476104736], [-0.03737959265708923], [-0.3955245018005371], [0.26750633120536804], [-0.4870208203792572], [-0.19262978434562683], [-0.3549192249774933]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_35775e7c00bb25289baa26afc893838e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3bd307b0a33d3c8eb57a86633fe32cc
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.45195016264915466], [-0.1353500485420227], [-0.4284287691116333], [-0.342902809381485], [-0.17490079998970032], [-0.24927039444446564], [-0.40733784437179565], [-0.1832345724105835], [-0.14042316377162933]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2b3e70fed94362f3b00ee418efb40dfe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30d54fd556c75f31938966f90b72b6ba
    def get_inputs(self):
        return [
            paddle.uniform([10, 240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6b5e807908cf7deb3db92fa30ebd38d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc7c3329af58c980f91d11faf6d58c9f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.2582399845123291]], [[0.27457892894744873]], [[0.43848156929016113]], [[0.03885175660252571]], [[0.03451291099190712]], [[0.36217233538627625]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([-10000000000.0], dtype='float32').reshape([1]),
            paddle.to_tensor([4.135169982910156], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_605b1a2da49f530f9572a32be942ff07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_605b1a2da49f530f9572a32be942ff07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c460a03d0543ade66507daec9ee08bd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc7c3329af58c980f91d11faf6d58c9f
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_46de2d8c4d297c229a90aa852e6d6b5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a757c2d16b1a5298599ec6a75f8b1ec2
    def get_inputs(self):
        return [
            paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_8749d633f341b11582eea651c82d2232(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.clip(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4a1e2afc93c7f66c762e931915d956ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8749d633f341b11582eea651c82d2232
    def get_inputs(self):
        return [
            paddle.uniform([10, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d893c921ae6f4b6a19a9cfb4a103fdcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_101546c092c13f4aeef5032773a9ee5c
    def get_inputs(self):
        return [
            paddle.uniform([10, 480], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2d385ab586ffa5729dba94b4518987bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2d385ab586ffa5729dba94b4518987bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cfd1eceabd2babd493d73e029c2f87d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc7c3329af58c980f91d11faf6d58c9f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-2.0], dtype='float32').reshape([1]),
            paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_eed16507ceb3f2fc1ac46a40968d31ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a757c2d16b1a5298599ec6a75f8b1ec2
    def get_inputs(self):
        return [
            paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_171a152c0d84c202521c518850d491f6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.clip(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 100, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a44314f9e413afb4b79ce839b0a8be38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_171a152c0d84c202521c518850d491f6
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 100, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0f8c350d8a41e2e7efec4ae781369d12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30d54fd556c75f31938966f90b72b6ba
    def get_inputs(self):
        return [
            paddle.uniform([171, 240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_387e6fe476d87d4b93ab486b92156895(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a757c2d16b1a5298599ec6a75f8b1ec2
    def get_inputs(self):
        return [
            paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f1ea529d4d2359c4f793f6916bc91d18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be491969ce50968f5ce95154e0c848d7
    def get_inputs(self):
        return [
            paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8f51e5ab87412f794f78a6adbdb26196(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8f51e5ab87412f794f78a6adbdb26196(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8f51e5ab87412f794f78a6adbdb26196(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8f51e5ab87412f794f78a6adbdb26196(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_97264156dc95c9322abc0e26c08db3e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_97264156dc95c9322abc0e26c08db3e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_97264156dc95c9322abc0e26c08db3e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_97264156dc95c9322abc0e26c08db3e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bf9c8bed0af493aad0d8f744709a5b61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bf9c8bed0af493aad0d8f744709a5b61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6795ab4e8c476fa9365ab7f8edca0811(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc7c3329af58c980f91d11faf6d58c9f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2fba229405fef0b206732276954abb97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2fba229405fef0b206732276954abb97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2fba229405fef0b206732276954abb97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2fba229405fef0b206732276954abb97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f2b7489b989f7a6b949f08a3aca8053a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30d54fd556c75f31938966f90b72b6ba
    def get_inputs(self):
        return [
            paddle.uniform([22, 240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2f0eac75f403b55095317dc88c1c2480(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2f0eac75f403b55095317dc88c1c2480(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2f0eac75f403b55095317dc88c1c2480(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2f0eac75f403b55095317dc88c1c2480(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1f1647a003d2e0910f50bb0ec0351d05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8749d633f341b11582eea651c82d2232
    def get_inputs(self):
        return [
            paddle.uniform([22, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f7353496236b1de2e627bec3f75e2e70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3bd307b0a33d3c8eb57a86633fe32cc
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.2856804132461548]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3870ca5a95b2793df3d1c9b2748e1005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3bd307b0a33d3c8eb57a86633fe32cc
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.13537442684173584]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_94fd814f55448699f8a5018058c54d8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3bd307b0a33d3c8eb57a86633fe32cc
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.03151401877403259], [-0.21845291554927826], [-0.24938222765922546], [-0.36716222763061523], [-0.03874947130680084], [-0.052606672048568726]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6cce45b8abca70f7aef9f9c4be68f08f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3bd307b0a33d3c8eb57a86633fe32cc
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.36969172954559326], [-0.1171245127916336], [-0.28271499276161194], [-0.1609843373298645], [-0.05140748620033264], [0.1431642770767212]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cea2fa6e027d8d34d6a5e92c62598093(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be491969ce50968f5ce95154e0c848d7
    def get_inputs(self):
        return [
            paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_f9dd61b76b80f0f4522a293b3513f7f0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.clip(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 672], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cf1ae5369f651b4bbf059bcda0c15d64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9dd61b76b80f0f4522a293b3513f7f0
    def get_inputs(self):
        return [
            paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_eed16507ceb3f2fc1ac46a40968d31ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a757c2d16b1a5298599ec6a75f8b1ec2
    def get_inputs(self):
        return [
            paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_64d0a0a8ece307d4498d0f6e58926e89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_64d0a0a8ece307d4498d0f6e58926e89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_64d0a0a8ece307d4498d0f6e58926e89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_64d0a0a8ece307d4498d0f6e58926e89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_076d2d4765ad9eb281271760246cabe6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_076d2d4765ad9eb281271760246cabe6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_076d2d4765ad9eb281271760246cabe6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_076d2d4765ad9eb281271760246cabe6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d861b2c90182741b043bbcb2b8b61dc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d861b2c90182741b043bbcb2b8b61dc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_74e648aa339e28f05f0602af754a8580(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc7c3329af58c980f91d11faf6d58c9f
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_53c96f7c67dd3a0183eb9e11c8384962(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_53c96f7c67dd3a0183eb9e11c8384962(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_53c96f7c67dd3a0183eb9e11c8384962(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_53c96f7c67dd3a0183eb9e11c8384962(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e05a741c85b07a7eaeb4c891c52fd952(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e05a741c85b07a7eaeb4c891c52fd952(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5a0c4c6f81c10ec5e43c2765c74311e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc7c3329af58c980f91d11faf6d58c9f
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ed295c6086af7cbf49ae020f8b5938af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_abda45cd897c6a197a0c6be62515e010
    def get_inputs(self):
        return [
            paddle.to_tensor(2434.0, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_22682838e4a4051900d7cf89987c00ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_22682838e4a4051900d7cf89987c00ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_beaf9f27a2de85b6554b6729bf5ce81f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc7c3329af58c980f91d11faf6d58c9f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cad6dbd7929691bd68b778d716d4580c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cad6dbd7929691bd68b778d716d4580c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cad6dbd7929691bd68b778d716d4580c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cad6dbd7929691bd68b778d716d4580c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2a3d79a58e255115a369aa053a54c61d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2a3d79a58e255115a369aa053a54c61d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2a3d79a58e255115a369aa053a54c61d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2a3d79a58e255115a369aa053a54c61d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f11498fe648747ff5930001e9290177b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3bd307b0a33d3c8eb57a86633fe32cc
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.15672597289085388], [-0.057970836758613586], [-0.169496551156044], [-0.07432708144187927], [-0.16166751086711884]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c0fcc7320a8eaf96ce3e5ed520dc7100(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3bd307b0a33d3c8eb57a86633fe32cc
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.31834909319877625], [-0.17892664670944214], [-0.33759501576423645], [-0.28139835596084595], [-0.17056933045387268]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_22bd32e3c50dcdcb8360f5f77066bdc1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.clip(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1248], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ce3e5b30e10be2eba812ea8e437542de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22bd32e3c50dcdcb8360f5f77066bdc1
    def get_inputs(self):
        return [
            paddle.uniform([1, 1248], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_35e0fa918b72f3c579941e75db76dcbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_35e0fa918b72f3c579941e75db76dcbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_35e0fa918b72f3c579941e75db76dcbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_35e0fa918b72f3c579941e75db76dcbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59d60c553114c3f17a793619dc134214(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_101546c092c13f4aeef5032773a9ee5c
    def get_inputs(self):
        return [
            paddle.uniform([171, 480], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_562b4c45c53016e05942b67a6ab8018e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_562b4c45c53016e05942b67a6ab8018e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_562b4c45c53016e05942b67a6ab8018e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_562b4c45c53016e05942b67a6ab8018e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_759f9c3e84ab4f74dbf332219d627bda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_759f9c3e84ab4f74dbf332219d627bda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_759f9c3e84ab4f74dbf332219d627bda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_759f9c3e84ab4f74dbf332219d627bda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a0d3a84eee93c4fa5a25953494dd2abc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8749d633f341b11582eea651c82d2232
    def get_inputs(self):
        return [
            paddle.uniform([145, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_264337af54ae85bac8f0ddde762cad6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_264337af54ae85bac8f0ddde762cad6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_11a8ab5218b9236e8da956b413b8102b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc7c3329af58c980f91d11faf6d58c9f
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_57bc1173634fd564a10bdc530e7e5bfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_57bc1173634fd564a10bdc530e7e5bfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_61522a4e6518c5606caddbe675995c11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc7c3329af58c980f91d11faf6d58c9f
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59a772500ad0b31b89390ab67a28024d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59a772500ad0b31b89390ab67a28024d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1c4ce7edd80d874a002d96dbfca6add3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc7c3329af58c980f91d11faf6d58c9f
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_92383bd9468f1cd3b00f9105c7403282(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.clip(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 156], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b58b369cb016af965177b524326888c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92383bd9468f1cd3b00f9105c7403282
    def get_inputs(self):
        return [
            paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_53c96f7c67dd3a0183eb9e11c8384962(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_53c96f7c67dd3a0183eb9e11c8384962(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_53c96f7c67dd3a0183eb9e11c8384962(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_53c96f7c67dd3a0183eb9e11c8384962(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e129881a4284943e8a925d51fbb8a529(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_309abcf95474782b38aa9a4ee326ecde
    def get_inputs(self):
        return [
            paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_405f6b733ded6bb29601f733f7244caf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_101546c092c13f4aeef5032773a9ee5c
    def get_inputs(self):
        return [
            paddle.uniform([22, 480], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_590705e3528e2e62463319e25fbe9aea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_101546c092c13f4aeef5032773a9ee5c
    def get_inputs(self):
        return [
            paddle.uniform([145, 480], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_121ff0a7924d14bcb1f539533085ca32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8749d633f341b11582eea651c82d2232
    def get_inputs(self):
        return [
            paddle.uniform([171, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_59610733337e0cc82872289db6bc156c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.clip(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 120], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8a22082107e0774f355771335bf19e62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59610733337e0cc82872289db6bc156c
    def get_inputs(self):
        return [
            paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cf1ae5369f651b4bbf059bcda0c15d64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9dd61b76b80f0f4522a293b3513f7f0
    def get_inputs(self):
        return [
            paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ec2caed04d9fc8e44bb53376f48f59e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3bd307b0a33d3c8eb57a86633fe32cc
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.21264928579330444], [-0.3496997356414795], [0.03776758909225464], [-0.09134560823440552]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_68b5aa455863f928dd66fb4a857244bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3bd307b0a33d3c8eb57a86633fe32cc
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.28148746490478516], [0.21245698630809784], [-0.0351884663105011], [0.1316526234149933]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8db86f5bca7f4150ba80ea8be4ae6755(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8db86f5bca7f4150ba80ea8be4ae6755(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_74e648aa339e28f05f0602af754a8580(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc7c3329af58c980f91d11faf6d58c9f
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_759f9c3e84ab4f74dbf332219d627bda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_759f9c3e84ab4f74dbf332219d627bda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_759f9c3e84ab4f74dbf332219d627bda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_759f9c3e84ab4f74dbf332219d627bda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2a59c2426f279e05c8e4901a7347b86c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2a59c2426f279e05c8e4901a7347b86c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2a59c2426f279e05c8e4901a7347b86c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2a59c2426f279e05c8e4901a7347b86c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_27d3703447166915569b80719d8821c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_27d3703447166915569b80719d8821c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_27d3703447166915569b80719d8821c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_27d3703447166915569b80719d8821c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_949cfec2a24621ab40bbea3dcbbeea6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_949cfec2a24621ab40bbea3dcbbeea6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0d0b686ba6834f9d56ccc3831bba1ea4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc7c3329af58c980f91d11faf6d58c9f
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_32f6825890ed1fa165c90cf99ac5a83d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.clip(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 624], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_690e6035a0a701341e08fabef2231d71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32f6825890ed1fa165c90cf99ac5a83d
    def get_inputs(self):
        return [
            paddle.uniform([1, 624], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_27d3703447166915569b80719d8821c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_27d3703447166915569b80719d8821c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_27d3703447166915569b80719d8821c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_27d3703447166915569b80719d8821c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22c0c1ee10fce4270dc54ca1bd45488a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()