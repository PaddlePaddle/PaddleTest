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


class TestPrimitiveOp_7791d38d822894833462d1f626aa82f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc7c3329af58c980f91d11faf6d58c9f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.09420384466648102]], [[0.4809618890285492]], [[0.04842865839600563]], [[0.32863229513168335]], [[0.21174632012844086]], [[0.3063952326774597]]], dtype='float32').reshape([6, 1, 1]),
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


class TestPrimitiveOp_669f51cf50653b87bf398c3e0836f887(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([1756, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_669f51cf50653b87bf398c3e0836f887(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([1756, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_a61a32ee2343f5804d6181e9f8f3aa4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3bd307b0a33d3c8eb57a86633fe32cc
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.153676837682724], [-0.451524943113327], [-0.4045058488845825], [-0.24965323507785797], [-0.24993786215782166], [-0.4155198633670807], [0.02123597264289856], [-0.11125600337982178], [-0.10108520090579987]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_587433d4d306b13964303b3b2f67f601(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3bd307b0a33d3c8eb57a86633fe32cc
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.43609243631362915], [-0.19829462468624115], [-0.05146096646785736], [-0.35840296745300293], [-0.10498392581939697], [-0.3461950421333313], [-0.10545861721038818], [-0.35533180832862854], [0.020342353731393814]], dtype='float32').reshape([9, 1]),
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


class TestPrimitiveOp_8810870448e8f4fc322c14ea6e363d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc7c3329af58c980f91d11faf6d58c9f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.02652212604880333]], [[0.28697821497917175]], [[0.2873174548149109]], [[0.231895312666893]], [[0.3593861162662506]], [[0.14348183572292328]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([-10000000000.0], dtype='float32').reshape([1]),
            paddle.to_tensor([4.135169982910156], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_238bae9663393aa899c58492f4522e6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([5551, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_238bae9663393aa899c58492f4522e6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([5551, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_7671564685db22fd4809cce579bad25e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([1769, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7671564685db22fd4809cce579bad25e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([1769, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_925fb2eddba5eed10593994e3ca3d010(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([1502, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_925fb2eddba5eed10593994e3ca3d010(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([1502, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_9924f4617c1c6c5f8ce6786c2ce0ec76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3bd307b0a33d3c8eb57a86633fe32cc
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.050895772874355316]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_453cc4da3f551198ab5a48c278ff12c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3bd307b0a33d3c8eb57a86633fe32cc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.028612107038497925]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d9ac731cef3936f4199183b5354ef47c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3bd307b0a33d3c8eb57a86633fe32cc
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.21635276079177856], [-0.19235265254974365], [-0.22706788778305054], [-0.2571583688259125], [-0.01586967706680298], [-0.07150772213935852]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e5cd9654ff5c3a84b7a6cdd35bbb0240(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3bd307b0a33d3c8eb57a86633fe32cc
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.46293872594833374], [-0.2054910957813263], [-0.15763702988624573], [0.06804439425468445], [0.02055343985557556], [0.04278743267059326]], dtype='float32').reshape([6, 1]),
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


class TestPrimitiveOp_1d1330807be688b9e6c82636b960d516(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([2080, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1d1330807be688b9e6c82636b960d516(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([2080, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_54cb3912384853a33ffa6ea40396018a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([4585, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_54cb3912384853a33ffa6ea40396018a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([4585, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_015fc79075e4c3f473970bb411baa7b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([1048, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_015fc79075e4c3f473970bb411baa7b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([1048, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_1589486883f796cb93725504d25c3e19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3bd307b0a33d3c8eb57a86633fe32cc
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.23078739643096924], [0.005014330148696899], [-0.2463493049144745], [0.2721625864505768], [-0.1740378588438034]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_177e6a189b76d3aab48bcfb285718bf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3bd307b0a33d3c8eb57a86633fe32cc
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.2742784917354584], [-0.16011777520179749], [0.1706799566745758], [-0.0793103277683258], [0.05108173191547394]], dtype='float32').reshape([5, 1]),
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


class TestPrimitiveOp_3a62f03e22c98d94cde2872cc4f8a0b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([2390, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3a62f03e22c98d94cde2872cc4f8a0b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([2390, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_80e0164ab5cc5589d07e7da1033c5d7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([3090, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_80e0164ab5cc5589d07e7da1033c5d7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([3090, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_099b13db0712596b41ed044393ae0c6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([3748, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_099b13db0712596b41ed044393ae0c6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([3748, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_59be8b77fce724e8c94e6122b2b230ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3bd307b0a33d3c8eb57a86633fe32cc
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.3661680817604065], [-0.21603424847126007], [-0.34582602977752686], [0.0254327654838562]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_15e7b89ce79b7013e7c36adb8adee4e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3bd307b0a33d3c8eb57a86633fe32cc
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.3337309956550598], [-0.17545807361602783], [-0.271816223859787], [-0.21927011013031006]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a27bf32f2d6a3b8cfd6a41f915030d8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([2031, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a27bf32f2d6a3b8cfd6a41f915030d8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([2031, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_f0e7197377e5fb06d027c76f4d5c54d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([4205, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f0e7197377e5fb06d027c76f4d5c54d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b376c15bd03abbb798b6d683a9d4a11
    def get_inputs(self):
        return [
            paddle.uniform([4205, 1], dtype='float32', min=0, max=0.5),
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