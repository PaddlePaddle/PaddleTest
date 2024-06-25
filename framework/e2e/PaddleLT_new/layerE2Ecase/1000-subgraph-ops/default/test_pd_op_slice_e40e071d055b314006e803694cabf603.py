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



class PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a38f63b028badbc2c8a10c7a3bc9cae0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 24, 36], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_7705834c68ee81af9711c7c14d255e09(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_25b5a3d4e577b3aedad58fac879de9df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 24, 36], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ed08d6fc3f0f1a30fd72743e2e258065(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 11, 6, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dc24889d1b56804f26652adf3b698086(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed08d6fc3f0f1a30fd72743e2e258065
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_2516227b434778854b031efd54dbdc00(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 11, 6, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f4d9029cf1f9d958e72f1452ab714b41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2516227b434778854b031efd54dbdc00
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6fa9e623542d7159a90b766612db416c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c03d990ad5abcbac4909f50bec10a775(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa9e623542d7159a90b766612db416c
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_7f536a76dcf4f51f6333d43b0e4def0e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 11, 64, 3, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4452ac6d6543c0f97d8b0bf8354ad06f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f536a76dcf4f51f6333d43b0e4def0e
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_3ae83dbb9b0c392e5629328963ddf0a8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 11, 64, 3, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2b58daa59b793ecc81b00009ea7e62f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ae83dbb9b0c392e5629328963ddf0a8
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f6644b566833a5533d3d4aa2c6fe5b8d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 11, 64, 3, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_925b38916d6ad99f414c76a943a0d17f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6644b566833a5533d3d4aa2c6fe5b8d
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_e7141c597d9cea06bb45553b2d876f07(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 4, None, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8c4cdc3370974242a2f0f5862c389df4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7141c597d9cea06bb45553b2d876f07
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_24bad0fe15c06b3cc5155f53241b481d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 4, None, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5b5d54cff5cbed5d28e9df5b8ed2ddc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bad0fe15c06b3cc5155f53241b481d
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_588f616b0b4d747e56904993f2f6e576(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 4, None, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b24ab67df9860efc8ffab95c2865604b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_588f616b0b4d747e56904993f2f6e576
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f8b48fd347ab5507461afa12e20e6055(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 3, 198, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d1f0eeb1c90895d4ccf7f8bd964e42bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8b48fd347ab5507461afa12e20e6055
    def get_inputs(self):
        return [
            paddle.uniform([3, 54, 3, 198, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_43d5250380b09b458be6303b68617b86(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 3, 198, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_19433ecd94eb43941be57a8e3cc05e5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43d5250380b09b458be6303b68617b86
    def get_inputs(self):
        return [
            paddle.uniform([3, 54, 3, 198, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_0561cc53150dac1d83dd15c31fb28e48(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 3, 198, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c1539aefda7e58b74ecbc7a76f00efee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0561cc53150dac1d83dd15c31fb28e48
    def get_inputs(self):
        return [
            paddle.uniform([3, 54, 3, 198, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_9d61ca7d6d47726805c5b9843e133c9a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 1960, 4, 16, 6], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_55a463a502ddb389ca37c800ebc1ef56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d61ca7d6d47726805c5b9843e133c9a
    def get_inputs(self):
        return [
            paddle.uniform([2, 1960, 4, 16, 6], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_9d0c3dfa7085c0e12acc8141f7d662eb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 1960, 4, 16, 6], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8850d02255ccf9d1f785770ae09da245(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d0c3dfa7085c0e12acc8141f7d662eb
    def get_inputs(self):
        return [
            paddle.uniform([2, 1960, 4, 16, 6], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_401f73950fbb98259dee1753c1387554(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 192, 7, 7], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e2952ec650172239bd46aa92a1b7582b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 192, 7, 7], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_48fa8e72448960f2f9aa0cb0545983f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 192, 7, 7], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_672d7ca116cb411bd493f55c9c11b420(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [3]
        input_2 = [4]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8eba71fafc1a34025f23a376ac9c6bc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 192, 7, 7], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_2034ba9c3a3eaad997aed58a1148d2ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 43, 6, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_33cc80e24536b0050747c4a06561ea29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2034ba9c3a3eaad997aed58a1148d2ee
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ed70f68191213dc4d797bf0a9a7c04a7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 43, 6, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d1df7ad0a114c8a11094141079e2916c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed70f68191213dc4d797bf0a9a7c04a7
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_701f4b217ff2cbb579c84255cd2fa82f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 112, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9b560dcfe4429e3b965cb06cbc1801fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 112, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_74f8b2c1311286b7269323625d23897e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 112, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9c39726fa74bf5c09b9b28c8fd26431e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 112, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_158cbb9c12207e29a0d866e46fe6a90e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 128, 16, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_43907d987ae539fef5885858293dbe9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 128, 16, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f526110456e71e1af522a319b6b6ae5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 128, 16, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_fca77c3c5cff818b9bf9df53882e0941(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 43, 64, 3, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b9ac9683b11e68c9ad2afa40b58069a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fca77c3c5cff818b9bf9df53882e0941
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_350b3c5854721752c12d8ec476d25687(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 43, 64, 3, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_da13c25bdef477617473da931a9f96ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_350b3c5854721752c12d8ec476d25687
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_e3b7850c3c2d74e65486900bcd3e43eb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 43, 64, 3, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b88fe161bc5fa0ad769fdc19f0807f59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3b7850c3c2d74e65486900bcd3e43eb
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_beac1656c826b0066c754a082454b7b3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_240f8fdcd81d26060cecc620d5d038cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3549, 76], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aab9931f936c693e403620dffbe559ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3549, 76], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_5be1605a4558c34e97c36d03a8a086d6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d688b5a8fab001484b3d7f694b6c2224(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5be1605a4558c34e97c36d03a8a086d6
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3549, 76], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a9bb945514388fdf19d22d657b49d5b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 576, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_96320d398a2c73126f7620ccb0b307f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 576, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_435fb04c6106b1c065700f701ed36469(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 576, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_74621263156663e6a582c99d3a0601fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 576, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a6a449874b4666f36c113d91931fe8bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 288, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47968e1b85c8dd757bc160c4a1594d9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 288, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_915c65a99847195ad64d35f02c7a5c09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 288, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2deff6b57447e1b15731911b5c3d400d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 288, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fb0e09c03bd483980724be19aed507d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 144, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_66e7d319c316ec920cfc575c4ce1697d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 144, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_adf2b48384d5f87dfacbc4e002f7e200(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 144, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bf43d20676b18f925b018c2312ec84e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 144, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_75041ef4cedab183bf63c9c20cf17af9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b3f9f4b604e208ac0c4df3a329fb519e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75041ef4cedab183bf63c9c20cf17af9
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.3036012053489685, 0.07570343464612961, 0.12798117101192474, 0.4362999498844147]], [[0.07301399111747742, 0.11000160127878189, 0.3936842083930969, 0.018207501620054245]], [[0.07366722077131271, 0.2992212474346161, 0.19797566533088684, 0.19333603978157043]], [[0.2978150546550751, 0.0893280953168869, 0.25166648626327515, 0.4248538613319397]], [[0.040806833654642105, 0.09959834814071655, 0.1797780990600586, 0.18371650576591492]], [[0.420791894197464, 0.14098724722862244, 0.007383337244391441, 0.42793917655944824]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d3300cf7fc66e5e1518bee0f4f2f2aa1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_de77e4f54966bc1ee21a22f13193b1e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3300cf7fc66e5e1518bee0f4f2f2aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.3036012053489685, 0.07570343464612961, 0.12798117101192474, 0.4362999498844147]], [[0.07301399111747742, 0.11000160127878189, 0.3936842083930969, 0.018207501620054245]], [[0.07366722077131271, 0.2992212474346161, 0.19797566533088684, 0.19333603978157043]], [[0.2978150546550751, 0.0893280953168869, 0.25166648626327515, 0.4248538613319397]], [[0.040806833654642105, 0.09959834814071655, 0.1797780990600586, 0.18371650576591492]], [[0.420791894197464, 0.14098724722862244, 0.007383337244391441, 0.42793917655944824]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_0bb3a180fd70a3d25cd873895bfc2ca2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f52390e27b9fa1cefdb187e36257faf1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bb3a180fd70a3d25cd873895bfc2ca2
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.3036012053489685, 0.07570343464612961, 0.12798117101192474, 0.4362999498844147]], [[0.07301399111747742, 0.11000160127878189, 0.3936842083930969, 0.018207501620054245]], [[0.07366722077131271, 0.2992212474346161, 0.19797566533088684, 0.19333603978157043]], [[0.2978150546550751, 0.0893280953168869, 0.25166648626327515, 0.4248538613319397]], [[0.040806833654642105, 0.09959834814071655, 0.1797780990600586, 0.18371650576591492]], [[0.420791894197464, 0.14098724722862244, 0.007383337244391441, 0.42793917655944824]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c8cf7775233c76dfa1df264c30034744(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [3]
        input_2 = [4]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a278ea28eb24380f7a15522097aac1f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8cf7775233c76dfa1df264c30034744
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.3036012053489685, 0.07570343464612961, 0.12798117101192474, 0.4362999498844147]], [[0.07301399111747742, 0.11000160127878189, 0.3936842083930969, 0.018207501620054245]], [[0.07366722077131271, 0.2992212474346161, 0.19797566533088684, 0.19333603978157043]], [[0.2978150546550751, 0.0893280953168869, 0.25166648626327515, 0.4248538613319397]], [[0.040806833654642105, 0.09959834814071655, 0.1797780990600586, 0.18371650576591492]], [[0.420791894197464, 0.14098724722862244, 0.007383337244391441, 0.42793917655944824]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bd08d0240c587485e356f4a44972e045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa9e623542d7159a90b766612db416c
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_09dab82ac7b0e033fcf23920fb2386db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 40, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1b76a0f60c6af100d5c616620fde024d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 40, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3590e3510eda04084056bd5dff4e35ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 40, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fcca43100726a1cf326beb772dcf5a39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 40, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1d3af4a78314b186009e67c13c3b94e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 80, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e88e5223ac61badfd01811edb077ab83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 80, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f9ac63c6b9fc7170af8bf471eb81d6ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 80, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5a6d7d3ceaff234e78bd4c2d8cc2056b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 80, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4e5d88f966540d35992e78a096a557a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1024, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_41f79bf1f53959068ac4be95e6098e7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1024, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_585cb61e763f860c3e88dfe3a190221f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 43, 12, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_abfd6a24f36167b0b8733c0958c82aaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_585cb61e763f860c3e88dfe3a190221f
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6bc328bdb4485d3323e516e45d5d3c91(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 43, 12, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5781ea54598707a9e0819da991070fe6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bc328bdb4485d3323e516e45d5d3c91
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8dc267f7eb3a5b386e0162d326a83d94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([128, 320, 8, 4], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_01847e9946cafecc0f16eabd46b8673a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([128, 320, 8, 4], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3aeab8513e619fb5477e485c95cb3607(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([128, 320, 8, 4], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e69df3bb9f1705d6aa9054fb56b170c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 49, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_708631d2c52fc041d9a4f7e8e2a4a03e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 49, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2140b29f056bd5562f348711d421a210(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 7, 7, 768], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b3845f5b5e9a8e067fabcb55d3305715(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e1b4723259a5c638f745a0a7c9d08c62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_498b5007ef7520f2b4874547f49d3bf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_95eb9f08cb919bcd467c0efe1ffe469f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3420248dc8bd3b6b4aec43e971f77594(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e9ba7735375c3614c30981c2c034a1f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_573ba45e304b10b85d841037c93f5eab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5948877d24dad30d0e92e67c76a1eaee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_0eb622d4f071e45d70c5247ad7d4e809(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_66283e53dd7f9d0caceecd10aa4e0255(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0eb622d4f071e45d70c5247ad7d4e809
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 128], dtype='int32').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_5d90941fe36f5b6c30bd60b852649221(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a0a320d6779c62b91668b4d5831cc63a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d90941fe36f5b6c30bd60b852649221
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 128], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_31b0eab2e866eac6445bf4e51bf9b82c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 43, 3, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cf66b7c1d799b01a492fb9ef416556c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31b0eab2e866eac6445bf4e51bf9b82c
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_0a9610c9f6d92ffe5fbc3a58f4bbeb7f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 43, 3, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c5092a8a9e3f97d7d9b9cbe0f373031a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a9610c9f6d92ffe5fbc3a58f4bbeb7f
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a9fa3706c870df8393e00c2098730115(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 80, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3befc7df95bc24f4b807787dc1b1fc53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 80, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3315e794edf71c66a283121c4911cab0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 80, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ffa9407036971a7dee936001fad138df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 80, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ad509540a0358cafd4a1f8327a88ecac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7141c597d9cea06bb45553b2d876f07
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1447a220c5b4a3991203290978fec636(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bad0fe15c06b3cc5155f53241b481d
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_53fd5e78583ceb3a2ec74943e05fc438(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_588f616b0b4d747e56904993f2f6e576
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8d12e7917490631061bb45b622050af7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 7581, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f147dcbaf6fb955b459d0bc86e2a66a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 7581, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4e7c637593121dd1274b664ac1ebfde7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5be1605a4558c34e97c36d03a8a086d6
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 7581, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3e9b2c6a5fbdbf2ba4b9af202b5a5efd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([528, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8cd1df9053c359cabe0808f4c93f8695(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([528, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0953a69b4297d8529993eac8c873ba68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([528, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_03e38065a51db5d4c7d97baca78fc937(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([528, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_170bd855adf1cd6ebcd276197308d112(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 25, 38], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6f1d474a1eb8906f8e68acc3b66fd6e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 25, 38], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6789b23ce0b408781573ecb6e1472f6d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 43, 16, 6, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7d7991fd1bdd58ad449840e161ca3407(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6789b23ce0b408781573ecb6e1472f6d
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ce2afdfe659129492b85bc8669422f17(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 43, 16, 6, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_65c17ffaca75053f9c0623d95b417b9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce2afdfe659129492b85bc8669422f17
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_cb8820a34599c9d3e36efac6c2599e01(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 43, 16, 6, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_80b0b20d551b813713ff0a9586a0ad17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb8820a34599c9d3e36efac6c2599e01
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e514543daee9bd5f75cf6f6aae092a4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([12, 288, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6a0b641095175364efacbc20a0129b42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([12, 288, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_530b8bbf1043a5bafba32c81d2320e0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5be1605a4558c34e97c36d03a8a086d6
    def get_inputs(self):
        return [
            paddle.to_tensor([12, 288, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8efb90ae160d19a147f4495fe84caa13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 320, 8, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ab0c6aa5e8a5cf44d8b4ac1802c771bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 320, 8, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_042eecf5972020d7988c398cbe1a77ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 320, 8, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_20f93970c9e30d37e5292ffd7f8906b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 4725, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4b6cfe93ff9821f0526524f8c14b41c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 4725, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9ce60ecec232d4d4b10231b07acbfa9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5be1605a4558c34e97c36d03a8a086d6
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 4725, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_63d1c63200382f575f456e75cb7ee2e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 160, 8, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7ec6fff72d5b4873945a056e0a4eca36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 160, 8, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_70b9f4b79eb3f32762d13d97f7014baa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 160, 8, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1186c8527bc751900e62dfd47420cd0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 577, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_343438911cd463bf18c783a55a206dff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 577, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_5a9a402c476057416e05e8445bee5c6d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 12, None, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a517daf68c3f311bedd5ee7b7c5335df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a9a402c476057416e05e8445bee5c6d
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 577, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_844745798fe7c273a18041fb38edd8d5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 12, None, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_07729c4272e8f69cfa049c81742d6c12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_844745798fe7c273a18041fb38edd8d5
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 577, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_0e5b40b387b25b5cb4c2c72230901880(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 12, None, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_00103408be0bde827232a92369837661(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e5b40b387b25b5cb4c2c72230901880
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 577, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2d9323f80102241d51cb75a788255c5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bd3e0ac6daf74cf16508d21e8432d16f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bc64beddd6e6349d3bdec3c3df82cc03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 14, 14, 384], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c7af791c2e7fc602115a48cf0cb233c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([64, 64, 32, 8], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_421c8769da744f99f81455503b1396fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([64, 64, 32, 8], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_08c6275939d1ce4581ac02141bae5217(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([64, 64, 32, 8], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6132ba09dec34dd2ccde008e471d7e3e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 10, 6, 197, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_057f939e2cd951175bca4ac444f565d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6132ba09dec34dd2ccde008e471d7e3e
    def get_inputs(self):
        return [
            paddle.uniform([2, 10, 6, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c7435eedfee8f1568a6fec91d73f9e20(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 10, 6, 197, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9688772a29ef5e709181ac755737c730(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7435eedfee8f1568a6fec91d73f9e20
    def get_inputs(self):
        return [
            paddle.uniform([2, 10, 6, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5ff94fd2410187558ec32827261e37dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([384, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_537c336ab6f00125be64719659042e4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([384, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_23e3825ee9df288f529c622f858572a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([384, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_aad68c99e51ca192c69158ad90966c12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([384, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_05bb76d5d7df7e3054170b98de550c64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 784, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_10c157ce4426ad9c7725c1976d286165(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 784, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5d84ecd8640ea6549ac1be79efd2d3c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 28, 28, 192], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_5774c85d985bb8efbed4d34c51563f66(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 11, 16, 6, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0f8bf3a5441aab06c3f75b789b78323c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5774c85d985bb8efbed4d34c51563f66
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_da3c07676a7811896c4ff0123349b281(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 11, 16, 6, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f5506c0d915c821685c7965bcd4f93d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da3c07676a7811896c4ff0123349b281
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d7d40eae8d58cd1f2c9b93e0def9a423(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 11, 16, 6, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_21cee6ff7a0192468f1e38cef5d6ad00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7d40eae8d58cd1f2c9b93e0def9a423
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_e921f5bdcd580bfe5e884931140cff35(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 1, 2, None, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8e51670fdf3bad2d680af0408598e4d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e921f5bdcd580bfe5e884931140cff35
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ba75ff8c2360a14f7fbb8f9a7799327e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 1, 2, None, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_69c07f7a5189441b5eb91d2699b0d347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba75ff8c2360a14f7fbb8f9a7799327e
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cf66b7c1d799b01a492fb9ef416556c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31b0eab2e866eac6445bf4e51bf9b82c
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c5092a8a9e3f97d7d9b9cbe0f373031a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a9610c9f6d92ffe5fbc3a58f4bbeb7f
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_712bae294b5e59dd1bc28933d81dba12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 128, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_662a062950e37e0096dde334d7bfd2ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 128, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d9273fb1b14fb2922113fdffd401e964(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 128, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_34f81d90256d82eb0113a46245cee406(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 43, 24, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4254cbc148a2da970b70287c30df9f70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34f81d90256d82eb0113a46245cee406
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_0f732c262cc7201569155ed9e64724c7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 43, 24, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6eae30b88503c29a64e0af2523426d03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f732c262cc7201569155ed9e64724c7
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_feb31c1bf0283600fb03bc15f016b99b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 8400, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_31be08fd63523b6ade436bb4472165a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 8400, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6f6873b3d8fccab5ebce48ae38b229ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5be1605a4558c34e97c36d03a8a086d6
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 8400, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ab4c98cabadf099f6502a9cbcd88beb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 112, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d42aad7ff31d8544508e43482e41253c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 112, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_72283116d2760e11a15078dfb7ba4d9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 112, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dca142edcfb7273ad8b6fc62eebe6cf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 112, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f7eee383aebdc339ed21c6140c58c2d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 20, 30], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5b5d39446067d17a3fa3f187cd783d0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 20, 30], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f386dcd6842f36ee38099997f1fa9561(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 40, 28, 28], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f2b633126e9606e88a5e8fce65909c85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 40, 28, 28], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_82f3aff4e588db8ce09c2627577bdebd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 40, 28, 28], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_780fe2aad984f74924baa3e2be4b25a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 40, 28, 28], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_63fef9bef26ee305abbcb21bdbf49855(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ceb9af5b5895933fdaba0f3f45858e72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_db886c31be33688c507c255a9e1f3adc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 56, 56, 96], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5b833363b6b7a3876bfdaf840fa2ae3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 24, 56, 56], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8680a607ac65bcdfb6dbff755e85c3b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 24, 56, 56], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_796751b47af3237d76813af60c179676(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 24, 56, 56], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_699a947a74ef5f9fbbe94c7fa3dffa6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 24, 56, 56], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c866077490580090f1b5bce55ad74239(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 160, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ebb75558732ccbc31d0f533891685fcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 160, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3e22b103d4656281f0b1de878c641457(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 160, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_45ebe77bda79f239716eecd8f5a5dfcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3c0a3c7996209c4124496c50fb0b071d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_481b9f44737925f2b4c78c2fbbbe7bac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_462d03f23230d0bec1a334568cf88b31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3420248dc8bd3b6b4aec43e971f77594(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e9ba7735375c3614c30981c2c034a1f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_573ba45e304b10b85d841037c93f5eab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5948877d24dad30d0e92e67c76a1eaee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3527076417ca3d3ed8fe735dece7aa6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3549, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5e987ba8b6dc4f1afdcb18409ce98652(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3549, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_eb2a7156e637b2eb20152e1e086409ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5be1605a4558c34e97c36d03a8a086d6
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3549, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3e9de423f3e45706c2b21b1ad1279937(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 768, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_25ce0a5b7a795e5dd72db9d10e591958(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 768, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c561ba07167a17fef0b2726851407257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 768, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_025731785402f39c9619ceed43d13a07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1025, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_537fbad1c5a3c249caf9c2ff251ddfaf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1025, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5ef9f7fbbdc7fefd86a73001a9c32089(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 64, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_60fb245263f439daa9bcf0e07721b958(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 64, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_e60e5e2fa1c3bd461ca599bda5acf597(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 2, None, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_158b62a2d3011cabb4fc6fdfd935993e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e60e5e2fa1c3bd461ca599bda5acf597
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 2, 640, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_cc7ef60272421c8ddf7e2eff9d4bec79(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 2, None, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a99f884ae5f67df291053984c5d3213c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc7ef60272421c8ddf7e2eff9d4bec79
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 2, 640, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_9476bfa93925fedbadd5877ad465cb5f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 2, None, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_06a1b0368b294fec7b0dcff5f1486c06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9476bfa93925fedbadd5877ad465cb5f
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 2, 640, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0ce23e3007adcb66803995816c946846(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8b48fd347ab5507461afa12e20e6055
    def get_inputs(self):
        return [
            paddle.uniform([3, 86, 3, 198, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a71b3f6af0c567d09fa85046f17d3ef4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43d5250380b09b458be6303b68617b86
    def get_inputs(self):
        return [
            paddle.uniform([3, 86, 3, 198, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4497f61d831b98772e444a2418b6b808(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0561cc53150dac1d83dd15c31fb28e48
    def get_inputs(self):
        return [
            paddle.uniform([3, 86, 3, 198, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b4ab6cf7b8f45ed95cabfcdee69a42fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b9ef552c68526db3a772ffddb0ca19a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_002b7ea362a6838c23ae6e4b89b39757(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 11, 3, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6dd38f510d350e37dee7a8970d9e5661(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002b7ea362a6838c23ae6e4b89b39757
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_1f06de965e7d9beeb22d0d1d763bc3e4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 11, 3, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_173d7ef543a0ad9d09c5ece4f62c29d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f06de965e7d9beeb22d0d1d763bc3e4
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b16c23de4f436fe6262c827dbf9196f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 192, 7, 7], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_786d14c3313677ef65c644a872b3402a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 192, 7, 7], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_edd04eb302512f6617c4b1d28870a706(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 192, 7, 7], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ea82fda7ed42a88d6eefb26f5cc52375(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 192, 7, 7], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6157f42a7dfc326ffad95cd5c498bbec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([20, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_211669e793dbfe27290a7545ff437101(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([20, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8f0f5f5e47b253c392f6ca623df0145e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([20, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_39ea10f9f519549155d01daa280efe8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([20, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_3d02c655902243039a959ad35762f637(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 43, 1, 24, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a8a358e5e591640f672b98c7aa4941af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d02c655902243039a959ad35762f637
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f7c0c5e55df0ed7cd7f87759e635c6a5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 43, 1, 24, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_219783ab6cc31f0b29ef874d30d9760d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7c0c5e55df0ed7cd7f87759e635c6a5
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_abbc97db6d9fc2a6ded58d73b33abcdb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 43, 1, 24, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b82b0c348147e5a33b85dbef49f71d69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_abbc97db6d9fc2a6ded58d73b33abcdb
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9067d91e63ab0ff27713804e596672ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 97, 97], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e3890a8dde9709b9775f8e51fef7db9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 97, 97], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_b679d5cbdde0b295fefb0eef552a102f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1, 2]
        input_2 = [98, 99]
        return paddle._C_ops.slice(input_0, [2, 3], input_1, input_2, [-1, -1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 104, 104], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fb4211eaa07b2657dcfa7abde0765d33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b679d5cbdde0b295fefb0eef552a102f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 104, 104], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 2], dtype='int64').reshape([2]),
            paddle.to_tensor([98, 99], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_b4f9bb3c27e2e7b6e2f6866642993d7d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 4312, 4, 16, 6], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_513c1ff0abe2e214b349aa4a0a3fd8d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4f9bb3c27e2e7b6e2f6866642993d7d
    def get_inputs(self):
        return [
            paddle.uniform([2, 4312, 4, 16, 6], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6c0de08d9e9121c033059815421c1065(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 4312, 4, 16, 6], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9405ede39fbe1804cb4211957c3c30aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c0de08d9e9121c033059815421c1065
    def get_inputs(self):
        return [
            paddle.uniform([2, 4312, 4, 16, 6], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e48abb19b056f00ae638d9b3fc250642(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ae05a0261d976cedbc899503f6e5e562(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b9ac9683b11e68c9ad2afa40b58069a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fca77c3c5cff818b9bf9df53882e0941
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_da13c25bdef477617473da931a9f96ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_350b3c5854721752c12d8ec476d25687
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b88fe161bc5fa0ad769fdc19f0807f59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3b7850c3c2d74e65486900bcd3e43eb
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2d53d46fafe1d2de74f67a4765518e76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 49, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ad5158b13bda221928852c0c1cfbebcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 49, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e24c7ff37a7095c18efc9a596b7fba21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 7, 7, 768], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9aaa2d995700c5acccfd9b5c1ea3a72a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 1024, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1e99b5ff035a6d1ee14400232ae53d5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 1024, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_847e30f035191b9ece57fbeb6fc98a1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 1024, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1c86d3b2aa8a51cb9e24561820061a24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 256, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9c02cce5450e8124df4e1fd17bec315b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 256, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1df617f32e94e890374cc7001ecca942(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 256, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ce957a40bceffc98c120fb31b94dd0c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([576, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a4b3b80d927437fe263b87040f3b1c18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([576, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_09788ecf8f369da1230d703c72b1f81c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([576, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ecc317ff2055e7209dc7f1eea0cce07b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([576, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6ed4e503897f847e6bee1e46be837fc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([22, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f890cd6cc694dce90d69f0cb6afbb196(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([22, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_64a3eb390ac84ad3f9064580aa2dac73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([22, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e835fb9c7113f85bec90377963175976(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([22, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_71876f5562f8e7665f1afa8866056f3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([96, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c1de7960278916f8edd655beec06eca2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([96, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5b018f9046ac7bea5c20db62c7726092(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([96, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_306ebf40310d35a12a8c181d1b61bd59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([96, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_520da385d06d0dbc88716bf84285dd60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([12, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_06480b1eec5c9f0f2aa7fb91d940284c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([12, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a08caf44d58c57cca47b25862e9f8ace(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([12, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_01a74446eb048f4bf30b2ca98e33c9c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([12, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ab95fad76d4e1a0259143246850244cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 256], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_65e4787f64248a6000fffd359c88190c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 256], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_56bb8fd3e090fce05f7fa47e69c5daeb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 8, None, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9536d9c9121723ad89448546139cd0ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56bb8fd3e090fce05f7fa47e69c5daeb
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 512, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6fd8deca636d9b26aed299873c7b4987(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 8, None, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_69bc550b355ff08ba747f84c6a364098(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fd8deca636d9b26aed299873c7b4987
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 512, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5ef9f7fbbdc7fefd86a73001a9c32089(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 64, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_60fb245263f439daa9bcf0e07721b958(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 64, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2b9d312a8a37e08fa02b56c887e5cc3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([6, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_20f1c3cdd7d4ea185d60ea26f54fddfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([6, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9dea4e8be785ccdc969e7b3e79d8a254(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([6, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_763d6fa155d263bfb129a977b01e04f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([6, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c605777dc616557595864ee0a0dca2a0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 11, 12, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_21e8cffff5520c95bd7e0a6db9af1d85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c605777dc616557595864ee0a0dca2a0
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_e8bcfc8e34240d0d4ece23a1b4ff4e37(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 11, 12, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ca746bdae046b750cb7040c4749e6b03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8bcfc8e34240d0d4ece23a1b4ff4e37
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e0f0f3d7cf6344fb34f803224c122f43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([960, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_679285c8d16109cfcf13255dc7730948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([960, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_beb8ebc19b43abefb50b582b5b9954ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([960, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fe640565db2def94fe8424b838f89109(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([960, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8c4cdc3370974242a2f0f5862c389df4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7141c597d9cea06bb45553b2d876f07
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5b5d54cff5cbed5d28e9df5b8ed2ddc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bad0fe15c06b3cc5155f53241b481d
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b24ab67df9860efc8ffab95c2865604b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_588f616b0b4d747e56904993f2f6e576
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3c5f0124a00b3f1d03ec267d21ed5d95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_63f4c6ac010529f0e4406084b7276105(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cea64254cce9c41484a053b594d3fb2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 256, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_930c363e49dfaa2a9a437868866d82a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 256, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7347d6c222a2a0c98d010b63c09cb0bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([2112, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e2ed2ae0f03170f069ae3330fbc958cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([2112, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a8bad4fb60c0334aa4797a45faf589f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([2112, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c4105994addcc4552f70852a43f328ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([2112, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0a95f8d0afddcf1c1803750e17bdfd6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 72, 28, 50], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_71d4cb5ba36575cd83009b65821b73f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 72, 28, 50], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_eb6a7accad01e81adb919c7e51ae8fa0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 72, 28, 50], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_781c1ad068c82457ba82f9cbf42c447f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 72, 28, 50], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2c4e77798620ac3edefcfc8cf836b164(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75041ef4cedab183bf63c9c20cf17af9
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.019471341744065285, 0.14261233806610107, 0.044832803308963776, 0.229569673538208]], [[0.027425510808825493, 0.4577380418777466, 0.26080331206321716, 0.39972877502441406]], [[0.4553568959236145, 0.10656984150409698, 0.1757422387599945, 0.39053621888160706]], [[0.315978467464447, 0.39828917384147644, 0.15780109167099, 0.4809509813785553]], [[0.30571866035461426, 0.38673222064971924, 0.2569504678249359, 0.05081148445606232]], [[0.06503966450691223, 0.32829776406288147, 0.35234102606773376, 0.19613081216812134]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a6aca7b0260cb1bcc0b8b34f6dfe4299(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3300cf7fc66e5e1518bee0f4f2f2aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.019471341744065285, 0.14261233806610107, 0.044832803308963776, 0.229569673538208]], [[0.027425510808825493, 0.4577380418777466, 0.26080331206321716, 0.39972877502441406]], [[0.4553568959236145, 0.10656984150409698, 0.1757422387599945, 0.39053621888160706]], [[0.315978467464447, 0.39828917384147644, 0.15780109167099, 0.4809509813785553]], [[0.30571866035461426, 0.38673222064971924, 0.2569504678249359, 0.05081148445606232]], [[0.06503966450691223, 0.32829776406288147, 0.35234102606773376, 0.19613081216812134]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4701525f3b0922d4454f7c8f56eb8ba4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bb3a180fd70a3d25cd873895bfc2ca2
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.019471341744065285, 0.14261233806610107, 0.044832803308963776, 0.229569673538208]], [[0.027425510808825493, 0.4577380418777466, 0.26080331206321716, 0.39972877502441406]], [[0.4553568959236145, 0.10656984150409698, 0.1757422387599945, 0.39053621888160706]], [[0.315978467464447, 0.39828917384147644, 0.15780109167099, 0.4809509813785553]], [[0.30571866035461426, 0.38673222064971924, 0.2569504678249359, 0.05081148445606232]], [[0.06503966450691223, 0.32829776406288147, 0.35234102606773376, 0.19613081216812134]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_188a8723a294e7c58eede4263c64a84c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8cf7775233c76dfa1df264c30034744
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.019471341744065285, 0.14261233806610107, 0.044832803308963776, 0.229569673538208]], [[0.027425510808825493, 0.4577380418777466, 0.26080331206321716, 0.39972877502441406]], [[0.4553568959236145, 0.10656984150409698, 0.1757422387599945, 0.39053621888160706]], [[0.315978467464447, 0.39828917384147644, 0.15780109167099, 0.4809509813785553]], [[0.30571866035461426, 0.38673222064971924, 0.2569504678249359, 0.05081148445606232]], [[0.06503966450691223, 0.32829776406288147, 0.35234102606773376, 0.19613081216812134]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e6125098a5d4a4a36d529145873c82fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 4116, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_339251ca1829c6f460d10b85a8ba18dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 4116, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c4af5a09faa42f17ba49cf9965ca62c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5be1605a4558c34e97c36d03a8a086d6
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 4116, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2d145eee10646f670ac75938f97aeb6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_70a73370ab479cadb37418edf2fcaa92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c50d688ad5a84d0bf6e51dfb7371209f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_139d3fe63bd1868c128d5d2ce3fc1233(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c50d688ad5a84d0bf6e51dfb7371209f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f244f5ebfbf27b14a8807111bf6940f0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0c5e007368ea5eb2755711a3ba0809cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f244f5ebfbf27b14a8807111bf6940f0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_991f5bcb4ed5ba071181d442b171e7b8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c81e8fa3c30170c5ed6d6eacb23e50d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_991f5bcb4ed5ba071181d442b171e7b8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_1bcf97e3cf372d17ea1336c88fc03b19(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [3]
        input_2 = [4]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cae1537e24b688f844e230a32e65c979(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bcf97e3cf372d17ea1336c88fc03b19
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d7718c1c4c903b9b7fd0dcea48aedeea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [4]
        input_2 = [5]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a9612cfa07bc97901ef4ce4afd32d2b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7718c1c4c903b9b7fd0dcea48aedeea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_fb3abc0afb3396ac6dfd43b449b65576(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [5]
        input_2 = [6]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aafd2c95baeb0e36d96815da874ce1ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb3abc0afb3396ac6dfd43b449b65576
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_2490b2bf008128c002edb57e2eaadf12(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [6]
        input_2 = [7]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_49d4d538c54505c4b5ef4e2065ed6ac9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2490b2bf008128c002edb57e2eaadf12
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_b04f86b910d4f541d6b976a8a4eff01d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [7]
        input_2 = [8]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1f1d069524ea3250a24c3dfb6d4cdeca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b04f86b910d4f541d6b976a8a4eff01d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_b61b8329738250e6625a6df0d0c1d432(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [8]
        input_2 = [9]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_054b11a956e9137390321ea0903438fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61b8329738250e6625a6df0d0c1d432
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f010398838e6f12d7087321b680afb3c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [9]
        input_2 = [10]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b10b9a85f8e4b16d6d89e551ae2ad566(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f010398838e6f12d7087321b680afb3c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d0d9fbd41c2980e17423e4b6c08bc20c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [10]
        input_2 = [11]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_52390b822e416a0dbc775173220756f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d0d9fbd41c2980e17423e4b6c08bc20c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d8ff8c8a6e9e7cd41791756582c92e68(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [11]
        input_2 = [12]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1102994e1d411dc33ca1cb8774a69da8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8ff8c8a6e9e7cd41791756582c92e68
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_a6ecdf730bab4b0880328b96b6893433(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [12]
        input_2 = [13]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_040e6fca214fd02468a764641ab18338(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6ecdf730bab4b0880328b96b6893433
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ba09e347c577b040962bc489d4a2a17e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [13]
        input_2 = [14]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_26aae7216994e9ca7fc0272425d97685(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba09e347c577b040962bc489d4a2a17e
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_13ca958a41e6ad054bfb95a191749bcf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [14]
        input_2 = [15]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_225112018f9bfd7f45c4b28668c0452d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13ca958a41e6ad054bfb95a191749bcf
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_fbae001c64314b77f09727e8c4ef38ba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [15]
        input_2 = [16]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_26da10fb4dc42ac086ce9120740b8b71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbae001c64314b77f09727e8c4ef38ba
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_2bb29e3c6ca3c565928633baf3cfe253(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [16]
        input_2 = [17]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_186a54dd8fcec3037f9f0d74d1844285(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bb29e3c6ca3c565928633baf3cfe253
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
            paddle.to_tensor([17], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_538c456786258f343e3444ea96ac160d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [17]
        input_2 = [18]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_407b9f79d8a624a89e09a1db7834bb79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_538c456786258f343e3444ea96ac160d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([17], dtype='int64').reshape([1]),
            paddle.to_tensor([18], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_bf3aba4c9e07690b4d4eb7608cbb152c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [18]
        input_2 = [19]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_79b2cf6902acc7c4197c730ac66fb5e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf3aba4c9e07690b4d4eb7608cbb152c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([18], dtype='int64').reshape([1]),
            paddle.to_tensor([19], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_cd9d6556e57a684715843791b69e9bb4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [19]
        input_2 = [20]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2f40d6cbbcddd1185cf69763aec97e99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd9d6556e57a684715843791b69e9bb4
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([19], dtype='int64').reshape([1]),
            paddle.to_tensor([20], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_e7ae8a68a19ea782546c855a6ae9465a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [20]
        input_2 = [21]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5f0cd496a43be4bcac1a756b3533164a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7ae8a68a19ea782546c855a6ae9465a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([20], dtype='int64').reshape([1]),
            paddle.to_tensor([21], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_dd53ffd94c495b6a8f4373f2091d353a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [21]
        input_2 = [22]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4364509dc53a7650628f0304d372f98f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd53ffd94c495b6a8f4373f2091d353a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([21], dtype='int64').reshape([1]),
            paddle.to_tensor([22], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ef42802da71621bfcdefd7d9739524f8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [22]
        input_2 = [23]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cad41b9c475dfb519a0e246ad30d03f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef42802da71621bfcdefd7d9739524f8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([22], dtype='int64').reshape([1]),
            paddle.to_tensor([23], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_97ab0da9a26db4efd90f911c7c1facca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [23]
        input_2 = [24]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ad5ff6601b9c3e7686524341e8a26826(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97ab0da9a26db4efd90f911c7c1facca
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([23], dtype='int64').reshape([1]),
            paddle.to_tensor([24], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_5d32024ff905de4193e3f7c885244169(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [24]
        input_2 = [25]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eaf1e9b721243a80f31bb057b07c811c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d32024ff905de4193e3f7c885244169
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([24], dtype='int64').reshape([1]),
            paddle.to_tensor([25], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_eeae17a07d3448904a393290deef4248(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [25]
        input_2 = [26]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_58601e1bfc1f1bda6f46e03ae8dfd49a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eeae17a07d3448904a393290deef4248
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([25], dtype='int64').reshape([1]),
            paddle.to_tensor([26], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ab044576f18c9f6a3591f67d696b42b7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [26]
        input_2 = [27]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_039578773b90204939ab7a1d05ae7b6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab044576f18c9f6a3591f67d696b42b7
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([26], dtype='int64').reshape([1]),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_405b6c899e117e16f09f8904ad06eb23(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [27]
        input_2 = [28]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_82cff67607c5804627454505cee001dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_405b6c899e117e16f09f8904ad06eb23
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
            paddle.to_tensor([28], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_3f89c5ae7d5b38014151d7c3df9f8517(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [28]
        input_2 = [29]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ca3483c24b4009c768799fdecdef8628(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f89c5ae7d5b38014151d7c3df9f8517
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([28], dtype='int64').reshape([1]),
            paddle.to_tensor([29], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_b8914a94f67339e378279ffc188aa081(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [29]
        input_2 = [30]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a2cdcdfd13aa82762b764461182dbae7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8914a94f67339e378279ffc188aa081
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([29], dtype='int64').reshape([1]),
            paddle.to_tensor([30], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6a617b3bf1c439ab74dd02f98fd804a1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [30]
        input_2 = [31]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_39f93f3d302a838965ed4972a2cc70cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a617b3bf1c439ab74dd02f98fd804a1
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([30], dtype='int64').reshape([1]),
            paddle.to_tensor([31], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6b7ef9792192c1dabf6549743038e27a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [31]
        input_2 = [32]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1517105864fdbb8a075557777d73572a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b7ef9792192c1dabf6549743038e27a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([31], dtype='int64').reshape([1]),
            paddle.to_tensor([32], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_1828c7590d2ea5145fa25d6edefa7cbd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [32]
        input_2 = [33]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ffc58da66151224a03efb86400bbc53b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1828c7590d2ea5145fa25d6edefa7cbd
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([32], dtype='int64').reshape([1]),
            paddle.to_tensor([33], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_01dc1fa4690516a04bdc86a0401f25de(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [33]
        input_2 = [34]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aa375b940064f6d3281ca44f47c5ef8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01dc1fa4690516a04bdc86a0401f25de
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([33], dtype='int64').reshape([1]),
            paddle.to_tensor([34], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_21cfd50e7c032b72885235de89fdf7f4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [34]
        input_2 = [35]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0dc9a8b363dfee2f490ff8b880dbb0b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21cfd50e7c032b72885235de89fdf7f4
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([34], dtype='int64').reshape([1]),
            paddle.to_tensor([35], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c058e105e18bd1382cdf2a47c4dc82a7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [35]
        input_2 = [36]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7a17f60011bd0d9767865f442e77d868(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c058e105e18bd1382cdf2a47c4dc82a7
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([35], dtype='int64').reshape([1]),
            paddle.to_tensor([36], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_335e738c56a5c9c2c5173e6e5b9a9680(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [36]
        input_2 = [37]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1ff6bd4d18fdbafaa692fb0169842243(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_335e738c56a5c9c2c5173e6e5b9a9680
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([36], dtype='int64').reshape([1]),
            paddle.to_tensor([37], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_b48651575d76e740060937b573657084(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [37]
        input_2 = [38]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_112ec0364a00547cce954adbb1a3f425(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b48651575d76e740060937b573657084
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([37], dtype='int64').reshape([1]),
            paddle.to_tensor([38], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_1e9a697df129d44e7064dc402831e209(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [38]
        input_2 = [39]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_15f566bfb59b911dc7ef5b9febf444d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e9a697df129d44e7064dc402831e209
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([38], dtype='int64').reshape([1]),
            paddle.to_tensor([39], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f833dacc2a2cda53a9c1a391b22b48e7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [39]
        input_2 = [40]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ea5a852ed626d32dc7d17245486b3be5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f833dacc2a2cda53a9c1a391b22b48e7
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([39], dtype='int64').reshape([1]),
            paddle.to_tensor([40], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_15e2fadd561f926054b9fed9090bf2bb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [40]
        input_2 = [41]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_21fd780c5a389a7f5378ce6f3af4edda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e2fadd561f926054b9fed9090bf2bb
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([40], dtype='int64').reshape([1]),
            paddle.to_tensor([41], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c12b01ab9b39699f8d5659a41477c84b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [41]
        input_2 = [42]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d6efc24b071f1eaee598f40f9050317f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c12b01ab9b39699f8d5659a41477c84b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([41], dtype='int64').reshape([1]),
            paddle.to_tensor([42], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_678e952c0b96c216ef68dce1c979149c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [42]
        input_2 = [43]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_92503129fbb38a1979511278e29a0a96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_678e952c0b96c216ef68dce1c979149c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([42], dtype='int64').reshape([1]),
            paddle.to_tensor([43], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_205188959c76d98034a5a1db343eda9f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [43]
        input_2 = [44]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2a36ac86f8529fcbd2a2b3b9b89d1161(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_205188959c76d98034a5a1db343eda9f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([43], dtype='int64').reshape([1]),
            paddle.to_tensor([44], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_7bcff84b449cafbe78a1b183848bc7cc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [44]
        input_2 = [45]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ee1131ed00eeda964a5b593fc1ef7019(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7bcff84b449cafbe78a1b183848bc7cc
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([44], dtype='int64').reshape([1]),
            paddle.to_tensor([45], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_8d3761e86a0c07288f5baa650982ea09(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [45]
        input_2 = [46]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e4a50e1b70015a083e943b87ba343583(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d3761e86a0c07288f5baa650982ea09
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([45], dtype='int64').reshape([1]),
            paddle.to_tensor([46], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_676e7e03ae22c6cdc112418879f40b40(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [46]
        input_2 = [47]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_58b0ba04aa5aa501f3bd1f7166eafe7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_676e7e03ae22c6cdc112418879f40b40
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([46], dtype='int64').reshape([1]),
            paddle.to_tensor([47], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f4ddecad5c14a73069eafc6cbbef39cc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [47]
        input_2 = [48]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a78b30878c198e45db61885c3244b581(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4ddecad5c14a73069eafc6cbbef39cc
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([47], dtype='int64').reshape([1]),
            paddle.to_tensor([48], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_5fb0d4a811b18d1435f0fd24cd1b88d6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [48]
        input_2 = [49]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d2718c666604c7b4a3fe3fce3b4ff924(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fb0d4a811b18d1435f0fd24cd1b88d6
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([48], dtype='int64').reshape([1]),
            paddle.to_tensor([49], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2d145eee10646f670ac75938f97aeb6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_70a73370ab479cadb37418edf2fcaa92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5083c2faf9281f5e9eb4dd671ead4b66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bd99ca05e439fd2bff9e31164f4b96c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3420248dc8bd3b6b4aec43e971f77594(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e9ba7735375c3614c30981c2c034a1f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_573ba45e304b10b85d841037c93f5eab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5948877d24dad30d0e92e67c76a1eaee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_63fef9bef26ee305abbcb21bdbf49855(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ceb9af5b5895933fdaba0f3f45858e72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_db886c31be33688c507c255a9e1f3adc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 56, 56, 96], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7e6861ffbb273105ece66781c9f42ce9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 960, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c5a6e5cbf72eebb675ff5ee654b51829(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 960, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2d18d3a7b480733b154b1160668ed031(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 960, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2e21e3efd53bb067dd0407607056cc80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 960, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ed90147862c35f31fdb7d25daf1d6b35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 480, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_545ed693207a55982c91e513a59467e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 480, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5ff01a9259fd0e41d1ea5edb6ab2a960(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 480, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_42118cfe0ec44d5afa0368f9177eeb0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 480, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1ff8d63b0a954a7e5acc8f51cc26a190(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 240, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_63e12c02c612370aa93cbc136c22f82d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 240, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_eaba5aa854aecc3daae47117f2f8f7bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 240, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dad9b29226b13b63307c2a46901719f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 240, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2104eeb9511bdd67e2c7238e9b437a18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 6069, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a718b0f6274ac65c09f18503d29c6de8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 6069, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1c82c850bb79473676286b995a3059c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5be1605a4558c34e97c36d03a8a086d6
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 6069, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f19710ccb932028912bdb6e0a88f08da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 784, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_35345ff30ab4be405e5b6e1a83bef7a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 784, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9e5a0a29506976b287da6563bd427e11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 28, 28, 192], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b16d73170f4eb3e262a2fb4d30fa4dc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 1, 1], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0f8bf3a5441aab06c3f75b789b78323c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5774c85d985bb8efbed4d34c51563f66
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f5506c0d915c821685c7965bcd4f93d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da3c07676a7811896c4ff0123349b281
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_21cee6ff7a0192468f1e38cef5d6ad00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7d40eae8d58cd1f2c9b93e0def9a423
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_13e74b35289198fd7209f62965ae5705(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d90941fe36f5b6c30bd60b852649221
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 8], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3410b5b85da993f34c3ddb67773e9070(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1025, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9b49448e09f38be337ae161958524829(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1025, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_7e4faf2f6e23ce0683ca1fbcdb17c227(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 6, None, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_11ed38df4d3d1ca5272c2df0152e6c6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e4faf2f6e23ce0683ca1fbcdb17c227
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 6, 1025, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_40165fb2fa79b5603a3e78db4b8689d7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 6, None, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5c4150be6365a9323588084709d0f0c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40165fb2fa79b5603a3e78db4b8689d7
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 6, 1025, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c5a1416a809bf9b5894d068cb38e65eb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 6, None, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d4a7e6b8d52d727e80cd3fbc41c91132(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5a1416a809bf9b5894d068cb38e65eb
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 6, 1025, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cf3fce9d89439b6750243299e405a456(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a62dbc811bdd7c8947c1be19ad90df30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_66d331d17ad788e44650a720c5ab4077(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2430af149d71be2636d18f349df04b27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 24, 56, 56], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3ee2d5f7630e80b41c13dc60ec286b48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 24, 56, 56], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ec65250c2745d1c012610ec3a8808ef8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 24, 56, 56], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_856ac618f3a3aa128f3f246e861b5a26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 24, 56, 56], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_586b9991aaa97df627564d7bd4fadc33(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 11, 24, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_abce228685aef4d1e7f3913b416e1b3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_586b9991aaa97df627564d7bd4fadc33
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c0f6b84f97f9d7836013abcf17d0cbc4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 11, 24, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5271895880c5be807982af6170e4318c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0f6b84f97f9d7836013abcf17d0cbc4
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b66222191f51c4db2e7aab2d9fc5392c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0eb622d4f071e45d70c5247ad7d4e809
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 64], dtype='int32').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7e18362674cdb90bb6a580d50e22a9db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d90941fe36f5b6c30bd60b852649221
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 64], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_243a7b04df911a1426bdf83a0fed456b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 1, 1], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b16d73170f4eb3e262a2fb4d30fa4dc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 1, 1], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2d9323f80102241d51cb75a788255c5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bd3e0ac6daf74cf16508d21e8432d16f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bc64beddd6e6349d3bdec3c3df82cc03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 14, 14, 384], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1bf3f900165485401772e20a91f05fba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0eb622d4f071e45d70c5247ad7d4e809
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512], dtype='int32').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a6ec13decd859bc4fab49d5590f2df1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d90941fe36f5b6c30bd60b852649221
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c2dafab74bb106dd6498703c0ab44bea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 1, 5, None, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e36fc7c08a0655455dc85be2e5354c9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2dafab74bb106dd6498703c0ab44bea
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 512, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f04c9c7baef74349d46f3d79a0fa5dc3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 1, 5, None, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ae152ba44bd3ccaadfba35c635ac13e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f04c9c7baef74349d46f3d79a0fa5dc3
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 512, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_cd665e5718849a8ba18aa0dd3e5d666f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4c072a2251c083c465006a55eadc5bf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd665e5718849a8ba18aa0dd3e5d666f
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 7], dtype='int64').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_35ea96ad8c5f9e2a52b3a7999ab47e93(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b97bec116a941515ce8a2989ee6bcf72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35ea96ad8c5f9e2a52b3a7999ab47e93
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 7], dtype='int64').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b471e21bbe6f9bcfc6df98bb6bbd2e57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1024, 256], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_64cb6dc4710d84b38ce764237f09f542(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1024, 256], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_76803b68b6eac7f9a4327f74c7cb0592(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56bb8fd3e090fce05f7fa47e69c5daeb
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 1024, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_75f18e449ef3b059a4c210d383db7268(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fd8deca636d9b26aed299873c7b4987
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 1024, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_087cde41e82f1ca169fe43ad4ed08f06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 64, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0e35a441c1a9b271f40c396585d6c1f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 64, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fe5a94fc1bdc44840611ca712a79da1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 64, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dc24889d1b56804f26652adf3b698086(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed08d6fc3f0f1a30fd72743e2e258065
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f4d9029cf1f9d958e72f1452ab714b41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2516227b434778854b031efd54dbdc00
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_116564bbaba5aa72f55bb30cd3a74a90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 464, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7ad4f6cebb0fdf6d955caf70d5420ecf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 464, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_af8ef1f32bf3790e74a37d040ae54e2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 464, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3ac4caf7afb13ea41d0709f9b0ffba61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 464, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a8a358e5e591640f672b98c7aa4941af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d02c655902243039a959ad35762f637
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_219783ab6cc31f0b29ef874d30d9760d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7c0c5e55df0ed7cd7f87759e635c6a5
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b82b0c348147e5a33b85dbef49f71d69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_abbc97db6d9fc2a6ded58d73b33abcdb
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_a4842f9df2594d2180f180bfc7607814(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 3, 197, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_275b6c0e3a2c219d5477266359c4be44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4842f9df2594d2180f180bfc7607814
    def get_inputs(self):
        return [
            paddle.uniform([3, 54, 3, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_be2707d6d85b33ef86b3ceba242e1961(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 3, 197, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_383f7580687a5e103ef19807cd299e03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2707d6d85b33ef86b3ceba242e1961
    def get_inputs(self):
        return [
            paddle.uniform([3, 54, 3, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_a35eadfbda236e342923f2b29a4c8b0f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 3, 197, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c84e4c889ce0776b9ddfe9061b45c6cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a35eadfbda236e342923f2b29a4c8b0f
    def get_inputs(self):
        return [
            paddle.uniform([3, 54, 3, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_45ebe77bda79f239716eecd8f5a5dfcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3c0a3c7996209c4124496c50fb0b071d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_481b9f44737925f2b4c78c2fbbbe7bac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_462d03f23230d0bec1a334568cf88b31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3420248dc8bd3b6b4aec43e971f77594(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_573ba45e304b10b85d841037c93f5eab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5948877d24dad30d0e92e67c76a1eaee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_45ebe77bda79f239716eecd8f5a5dfcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_481b9f44737925f2b4c78c2fbbbe7bac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_462d03f23230d0bec1a334568cf88b31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_7949ff795bd249c5d4af82cbc39ff51a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 1, 1, None, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8c3025be21400948b698c95b80bfc3f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7949ff795bd249c5d4af82cbc39ff51a
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 1024, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_8e08bb01bfe322016e7a4cfc2ee9195a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 1, 1, None, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1937fb2ef50e2b31b4bc2c915682f5ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e08bb01bfe322016e7a4cfc2ee9195a
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 1024, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c03d990ad5abcbac4909f50bec10a775(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa9e623542d7159a90b766612db416c
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7d7991fd1bdd58ad449840e161ca3407(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6789b23ce0b408781573ecb6e1472f6d
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_65c17ffaca75053f9c0623d95b417b9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce2afdfe659129492b85bc8669422f17
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_80b0b20d551b813713ff0a9586a0ad17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb8820a34599c9d3e36efac6c2599e01
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2ade13a5da44aec6688253503a1fc55a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 256, 8, 16], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c314415db5ad2182e29ea1f5b02f9979(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 256, 8, 16], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a1310fbbfc8a4e4009a42945a81f313f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 256, 8, 16], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0ac88a7ec80e0fe7efe049b871f1a589(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bb80b71efbe8cb6e8420d666d774e8b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_12f2a82b10c7b73cc1015f124ddec1cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 14, 14, 384], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_28e787e88c110ede608eaaf8dd41c851(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_57642223e1df635fe6f393c90d84199b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2e2b53072aaaa8c834a809b38b9867cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e6bac0fda91a7d1122c6e9e156631a6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cdcbd002e717113fef061df3d68a2e35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 144, 14, 25], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8a3ae13661ac5e71976d7833be57d161(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 144, 14, 25], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a084f62da0bfb6cbb0745563787d0bd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 144, 14, 25], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_06108a1d300b0a5f1d7472418479c517(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 144, 14, 25], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6dd38f510d350e37dee7a8970d9e5661(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002b7ea362a6838c23ae6e4b89b39757
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_173d7ef543a0ad9d09c5ece4f62c29d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f06de965e7d9beeb22d0d1d763bc3e4
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2a2d5dc75d690812000f545f5fc09766(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 128, 1, 1], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_476a1860d55b6cf419d2ab86ecf868ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 128, 1, 1], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c7cbefce2c84a8c17dd924eddac2e580(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0c0d65ac4d0e3a2141a2a690f2d7ca03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2504c5aa95b2942d2fd7bec9b55a934b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 56, 56, 96], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ad509540a0358cafd4a1f8327a88ecac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7141c597d9cea06bb45553b2d876f07
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1447a220c5b4a3991203290978fec636(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bad0fe15c06b3cc5155f53241b481d
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_53fd5e78583ceb3a2ec74943e05fc438(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_588f616b0b4d747e56904993f2f6e576
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_89a973be4b6d2c0af1abd9223ffbc7aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 1, 1, None, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2aa8fbc86d91791fb197cb12a593fa56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89a973be4b6d2c0af1abd9223ffbc7aa
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_8c64e093bff901e5ad956ab61e94ebb9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 1, 1, None, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_449da2597cade97e402eb6ca3c9789f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c64e093bff901e5ad956ab61e94ebb9
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_abce228685aef4d1e7f3913b416e1b3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_586b9991aaa97df627564d7bd4fadc33
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5271895880c5be807982af6170e4318c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0f6b84f97f9d7836013abcf17d0cbc4
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b66222191f51c4db2e7aab2d9fc5392c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0eb622d4f071e45d70c5247ad7d4e809
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 64], dtype='int32').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7e18362674cdb90bb6a580d50e22a9db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d90941fe36f5b6c30bd60b852649221
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 64], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1dcb5b5f4bd6b51c638fd99d805df3cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e60e5e2fa1c3bd461ca599bda5acf597
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 2, 200, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_865208c0877ae2c4009eab96f7e67d1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc7ef60272421c8ddf7e2eff9d4bec79
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 2, 200, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8d716f6d139a407f12c14dd7ce51d5d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9476bfa93925fedbadd5877ad465cb5f
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 2, 200, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c8d0254e72891d474680dec886c9b22a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 9261, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c600b5fbfe66720096e0940d600c3549(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 9261, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1e33a6be289f7ad002bd0752021e9535(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5be1605a4558c34e97c36d03a8a086d6
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 9261, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_accf37a0f5a564156a0890c1a01c8d1e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_335ed37eed6ef26453faf386a08302f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_accf37a0f5a564156a0890c1a01c8d1e
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_3ba9e3ecc7aefd07a12a926630167a2b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_04063f5b20a9d3ad33be9343fac355e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ba9e3ecc7aefd07a12a926630167a2b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_41aa0f99178c08b47f1ea24271aae0de(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6510abb8c951684aa582aceb2bb4c35f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41aa0f99178c08b47f1ea24271aae0de
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_8991cc386dfe6077ba1e45fea25759c9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [3]
        input_2 = [4]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_221e5a3f4ae8e8f5ae9efa12fbda8733(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8991cc386dfe6077ba1e45fea25759c9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_86e923be65b1a3560154e7d03e4a44ed(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [4]
        input_2 = [5]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_99048a012d7f06bfc9d94ff2008c52c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86e923be65b1a3560154e7d03e4a44ed
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4f6f8f5fdfcf4fd8fc4d6731a4502567(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [5]
        input_2 = [6]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b1db6130bed42b22204acf59cb0a4bec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f6f8f5fdfcf4fd8fc4d6731a4502567
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_a4c931bd587049811f872a1f5e314125(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [6]
        input_2 = [7]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0b1716d99339e3be4e0de6f93b1fe7e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c931bd587049811f872a1f5e314125
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f8f675979b95c010b78ef4e55ad8ec33(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [7]
        input_2 = [8]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ae727e53dc00b7c9f7d85525079225c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8f675979b95c010b78ef4e55ad8ec33
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d079da2c40e04afcee7253847677c69c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [8]
        input_2 = [9]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_596126d954845cf97b8e306135204d06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d079da2c40e04afcee7253847677c69c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_50d446c5fdf3f32ad804b5fda30dcd16(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [9]
        input_2 = [10]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_208fbfb322e92799cd74413640ddd3c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50d446c5fdf3f32ad804b5fda30dcd16
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_855b9b2db82e33602b1fbbd78e5ffe02(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [10]
        input_2 = [11]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4a94f32fe626913ad162a1a435fe6a00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_855b9b2db82e33602b1fbbd78e5ffe02
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f80b9a16ea71fa6c0f99cc34936b0bae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [11]
        input_2 = [12]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c4bf34107312416f43838dcda3139ffc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f80b9a16ea71fa6c0f99cc34936b0bae
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_9960c973a9f0a71a72d24c83dab22c34(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [12]
        input_2 = [13]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_70fbd270c913fa845e68ed3041754571(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9960c973a9f0a71a72d24c83dab22c34
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_07d8806293c26737ea42be3fe3f97f3a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [13]
        input_2 = [14]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4c56cd4dcfd019704e66a7ce0297da5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07d8806293c26737ea42be3fe3f97f3a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4c8ad674787669b9ae6184a05c893d7e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [14]
        input_2 = [15]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5a2266f09e33dec65453877dc9949ffe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c8ad674787669b9ae6184a05c893d7e
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_979215e17f78b02901f27edd6077d7cc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [15]
        input_2 = [16]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_756112f67dd8d3e750f97400ff066ae5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_979215e17f78b02901f27edd6077d7cc
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_09968beb4f845d6536da02733e4a8dd2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 22, 6, 197, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7522c278a86c46eb551ed843a4dd7e7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09968beb4f845d6536da02733e4a8dd2
    def get_inputs(self):
        return [
            paddle.uniform([2, 22, 6, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d1b5e6ee95017df9f16cf6f6dea0ecc4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 22, 6, 197, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_91c434587cdcb64a2ba84d2740fb9816(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1b5e6ee95017df9f16cf6f6dea0ecc4
    def get_inputs(self):
        return [
            paddle.uniform([2, 22, 6, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f84333e6e99f9aa00c125d2ca0e3b1f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([240, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_20a13f2e62eba7d744af006663dd9201(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([240, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4c3860eaac6e16e810d256e6f4bdde09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([240, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3ce584cae2535af6d0169469a3b363a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([240, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_06eb1e05545a4f11e41aa4eda156c016(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8a217372d55d7cf2f2660f49abd8f587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5d8bc202feea2e66e5011924a8d4da6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ceb672f4b155d28290653db8b1954485(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b4ab6cf7b8f45ed95cabfcdee69a42fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b9ef552c68526db3a772ffddb0ca19a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b4ab6cf7b8f45ed95cabfcdee69a42fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b9ef552c68526db3a772ffddb0ca19a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b4ab6cf7b8f45ed95cabfcdee69a42fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b9ef552c68526db3a772ffddb0ca19a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3069f25e05baf30c50edd84f0ec37336(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2048, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_27f3dd85333221b8e7dec868bb05fecf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2048, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f19710ccb932028912bdb6e0a88f08da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 784, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_35345ff30ab4be405e5b6e1a83bef7a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 784, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9e5a0a29506976b287da6563bd427e11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 28, 28, 192], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4a0860ee6052799856e887fd6f0035ce(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 43, 4, 12, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4a5aa1e9e855b313e7dfd08945c8254f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a0860ee6052799856e887fd6f0035ce
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_11dd6f3e9974b386807b153b137e1deb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 43, 4, 12, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9103fa34d6e53149f268ed877446f4c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11dd6f3e9974b386807b153b137e1deb
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d89033c749ca79fca5fa39bfbe49066b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 43, 4, 12, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_affa6f823ad1609a19dc8ecf4eb89f58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d89033c749ca79fca5fa39bfbe49066b
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9078b305e3c61e30bdbc6157c9818458(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e921f5bdcd580bfe5e884931140cff35
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1b172db6854839c68c8c2a6492d325c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba75ff8c2360a14f7fbb8f9a7799327e
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d3b29ec677db63700a1aa247c6e730ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9edf0c1faef62c42fca9445dd87d5c76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a667065254cc2e89f5792564f423af52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c7c5d95dc4e9c5220c3f155f0c956493(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8e3711bc0cb383e65c359531f4f7b5bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dbc46223470f887b7559b5dfb4c80858(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_16f0b1357348e5f5d61eb4c5328634e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e36552655b98f3c2b35f066ea0a458f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6dae889aa801405e76736a68295b3d62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_158abe2dbee7d90a66ac68d5618b893c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_03f8a029083cadf2af6fbdafe71c9443(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_43f31c9577cfd11c797b805c97168ab5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_30ee17eceec6423f9212c309202b01d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 16, 16], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4c12f294b3460d3265d926e5e0449156(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 16, 16], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ff20a4f7796aa96947de4cedb123cbeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 16, 16], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a948f294a2a35a80c6b7a2d275733463(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 16, 16], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_18252fd3522ba8cf388849f25a7b86b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 8, 8], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_638d2b28f0f5e3a9ab18103b24595ee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 8, 8], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_46196c30228a61cb110b4b26859c69a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 8, 8], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_50f8e25df8286406031c774d850f07d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 8, 8], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d1fea7006dadca88b2de49f6fb0e33b3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 1, 5, None, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_82018286b599a4cc2ae6b152f4b10d2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1fea7006dadca88b2de49f6fb0e33b3
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6c9f45b1f2c221a954d9c7a2b9fa31a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 1, 5, None, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_869dfd2855a2504ee46f0ba6bece76e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c9f45b1f2c221a954d9c7a2b9fa31a6
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a355f289f32e1cfcf010a95dc797139c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2100, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fe4e261ff9d5cd169c2ec8571eef1cc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2100, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4ce3920cb4f87f34d8bbfff60da9e6e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5be1605a4558c34e97c36d03a8a086d6
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2100, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b4ab6cf7b8f45ed95cabfcdee69a42fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e77d88aa0328c6db51dc9230b7bde9aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_67f3523817d15d364f5c04c69c13fd6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cf3fce9d89439b6750243299e405a456(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_01f837ac83a5e47d515fea9deef58155(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cf3fce9d89439b6750243299e405a456(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_01f837ac83a5e47d515fea9deef58155(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cf3fce9d89439b6750243299e405a456(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_01f837ac83a5e47d515fea9deef58155(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d0f9ef84cbb945a3e703f7017a9397cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2048, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_049b9f1ad52874334ce31b2408fa1c5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2048, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0ac88a7ec80e0fe7efe049b871f1a589(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bb80b71efbe8cb6e8420d666d774e8b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_12f2a82b10c7b73cc1015f124ddec1cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 14, 14, 384], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1cad54b69a7ea41b7988c260943561de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 15, 25], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d872211d511147ac5c6347b6f432f338(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 15, 25], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2d9323f80102241d51cb75a788255c5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bd3e0ac6daf74cf16508d21e8432d16f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bc64beddd6e6349d3bdec3c3df82cc03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 14, 14, 384], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_2722fed3dcb110f53c4d0828102e2ccb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 11, 1, 24, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b7a618a1dd3be80bca2b71eb0251ff8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2722fed3dcb110f53c4d0828102e2ccb
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_2fda4be259bf0b1d0224177868cc3b76(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 11, 1, 24, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_894663f4bbd77fff0c26d0f8ef0469d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2fda4be259bf0b1d0224177868cc3b76
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f76bf91a78de067392bac5a7d5b1c3c1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 11, 1, 24, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c9d0c9a1c8c2d8a4f49fca62c221d1af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f76bf91a78de067392bac5a7d5b1c3c1
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_8828536c000780b3f3760fa706c88883(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 11, 4, 12, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_127e87a2ce6cba675c7626727bfa01c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8828536c000780b3f3760fa706c88883
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_0dad2bbcf372dcff3d8b071401bb8d70(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 11, 4, 12, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3c93b4e933591d40e2042ea0e9b115a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0dad2bbcf372dcff3d8b071401bb8d70
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_3ff084212ec85ab9699df8dda76487c0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 11, 4, 12, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e737b84741744c77dd15a73c2b3c519d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ff084212ec85ab9699df8dda76487c0
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b7a618a1dd3be80bca2b71eb0251ff8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2722fed3dcb110f53c4d0828102e2ccb
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_894663f4bbd77fff0c26d0f8ef0469d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2fda4be259bf0b1d0224177868cc3b76
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c9d0c9a1c8c2d8a4f49fca62c221d1af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f76bf91a78de067392bac5a7d5b1c3c1
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_025731785402f39c9619ceed43d13a07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1025, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_537fbad1c5a3c249caf9c2ff251ddfaf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1025, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_16475dadb764e66c2574ddcbcba41151(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a9a402c476057416e05e8445bee5c6d
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 1025, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_98975d5e20314b8df05a6a895caa8089(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_844745798fe7c273a18041fb38edd8d5
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 1025, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_36742305681c01e1b5c5adffe37856ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e5b40b387b25b5cb4c2c72230901880
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 1025, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7fda02f4aa92be8ae0d50e1f20a9ffa1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([44, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b5032ccd03465c6910db2e42dc54dcd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([44, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d9d0cd091575c964a23cd500cb42595d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([44, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a82aea4a0165d358e51919e916f00785(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([44, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_127e87a2ce6cba675c7626727bfa01c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8828536c000780b3f3760fa706c88883
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3c93b4e933591d40e2042ea0e9b115a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0dad2bbcf372dcff3d8b071401bb8d70
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e737b84741744c77dd15a73c2b3c519d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ff084212ec85ab9699df8dda76487c0
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a7235bb83c416012969c0ea2afe97ab2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 80, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1fc18dba166bac54c600621dff5a0be8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 80, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8f1350a4128e346c4f94648b6c8d92fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 80, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f502952fa90170e026e8292596be7618(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 80, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c7cbefce2c84a8c17dd924eddac2e580(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0c0d65ac4d0e3a2141a2a690f2d7ca03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2504c5aa95b2942d2fd7bec9b55a934b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 56, 56, 96], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bd08d0240c587485e356f4a44972e045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa9e623542d7159a90b766612db416c
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b4ab6cf7b8f45ed95cabfcdee69a42fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b9ef552c68526db3a772ffddb0ca19a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e77d88aa0328c6db51dc9230b7bde9aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_67f3523817d15d364f5c04c69c13fd6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cd839f5232bbcc9971eae6381c38dc95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1024, 512], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5ff17225dd1054fdc1b3ce11e006a93c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1024, 512], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d2f46d666d69da7c25d0b837a2576f07(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 8, None, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a7cae60ed992a021de59278bc83e1a17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2f46d666d69da7c25d0b837a2576f07
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_09947e9d9cc49cbf1ba92cb97413c652(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 8, None, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dd3e26c6c12dc59e4a9a1508ae648fdf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09947e9d9cc49cbf1ba92cb97413c652
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0b2d962a8b574413b56692042ea50e83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 11109, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2dba0367a49c50115a2617086f9ef6cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 11109, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dabc9b26a84fb511770f29ea59680191(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5be1605a4558c34e97c36d03a8a086d6
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 11109, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4452ac6d6543c0f97d8b0bf8354ad06f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f536a76dcf4f51f6333d43b0e4def0e
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2b58daa59b793ecc81b00009ea7e62f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ae83dbb9b0c392e5629328963ddf0a8
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_925b38916d6ad99f414c76a943a0d17f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6644b566833a5533d3d4aa2c6fe5b8d
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e69df3bb9f1705d6aa9054fb56b170c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 49, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_708631d2c52fc041d9a4f7e8e2a4a03e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 49, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2140b29f056bd5562f348711d421a210(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 7, 7, 768], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4a5aa1e9e855b313e7dfd08945c8254f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a0860ee6052799856e887fd6f0035ce
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9103fa34d6e53149f268ed877446f4c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11dd6f3e9974b386807b153b137e1deb
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_affa6f823ad1609a19dc8ecf4eb89f58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d89033c749ca79fca5fa39bfbe49066b
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2d53d46fafe1d2de74f67a4765518e76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 49, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ad5158b13bda221928852c0c1cfbebcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 49, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e24c7ff37a7095c18efc9a596b7fba21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 7, 7, 768], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4254cbc148a2da970b70287c30df9f70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34f81d90256d82eb0113a46245cee406
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6eae30b88503c29a64e0af2523426d03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f732c262cc7201569155ed9e64724c7
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_80df4d3386571f102756e05ea1fd6721(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8f82c98a9e6ff9d8339987da17116f29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8e3711bc0cb383e65c359531f4f7b5bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dbc46223470f887b7559b5dfb4c80858(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_073e36acbcdb9ec0449cfd5a5ff02a42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([144, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_360ce671ae1ad4d44b02872b67bc60ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([144, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9180805affca56a36fff3bd7b340671d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([144, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_26957689abf986fe381d0b79ffa1fdf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([144, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7ccdc974ed9c50348c51b32f2ad9b9c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1fea7006dadca88b2de49f6fb0e33b3
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ac054b8964d82ac3944210059157d947(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c9f45b1f2c221a954d9c7a2b9fa31a6
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a6dc02c0e7d0768bc82931f6700d42af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2dafab74bb106dd6498703c0ab44bea
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 1024, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_276a8509d435bb067b21cfe347a7409d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f04c9c7baef74349d46f3d79a0fa5dc3
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 1024, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_09dab82ac7b0e033fcf23920fb2386db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 40, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1b76a0f60c6af100d5c616620fde024d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 40, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3590e3510eda04084056bd5dff4e35ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 40, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fcca43100726a1cf326beb772dcf5a39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 40, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1d3af4a78314b186009e67c13c3b94e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 80, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e88e5223ac61badfd01811edb077ab83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 80, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f9ac63c6b9fc7170af8bf471eb81d6ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 80, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5a6d7d3ceaff234e78bd4c2d8cc2056b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 80, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5bcb842b4d88cbd4f630c5aa21b71141(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 160, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6c23800a0e97e2c417e6cf37e4d4f1cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 160, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6119a530cf0f4e9ffcce071d91173b71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 160, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_848d37d4c965653af1a648accc351ed6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 160, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_34643cf69a53d1139a77bf522fbb3a92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 384, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6f8f00a49dedeb47a66c2028e10de5a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 384, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8439224288e6d62a2525d59dc2cc605c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 384, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3cf59a5c6a22c1657bb798b25a6573bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 384, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e48abb19b056f00ae638d9b3fc250642(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ae05a0261d976cedbc899503f6e5e562(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2d124b29d4ff730b087dbc555acba835(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_be4052e542e530e353ee75d967c82ea1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c3f166b53b15c38cd2f3e5f435b280f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 96, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_57faf816970dff51cbffd0a852f72665(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 96, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_eea6ec688575650d7397aba89b9e077c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 96, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e1f9833e2416287721ded6b7c0554c6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 96, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_abfd6a24f36167b0b8733c0958c82aaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_585cb61e763f860c3e88dfe3a190221f
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5781ea54598707a9e0819da991070fe6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bc328bdb4485d3323e516e45d5d3c91
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e38694fcb130dfb19c4dc9e1b517838f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 512], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_74582f3f1148d31ea11973f6ab8f0be4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 512], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8e7f89708f67e37372d24ccc54809905(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2f46d666d69da7c25d0b837a2576f07
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e41407caab1d49c293917b72d02cd4b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09947e9d9cc49cbf1ba92cb97413c652
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_05bb76d5d7df7e3054170b98de550c64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 784, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_10c157ce4426ad9c7725c1976d286165(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 784, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5d84ecd8640ea6549ac1be79efd2d3c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 28, 28, 192], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f28bf23e72c261409f427049fc6b98d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1024, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_89b02f39bc0f84684967ab3521ef38bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1024, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_33cc80e24536b0050747c4a06561ea29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2034ba9c3a3eaad997aed58a1148d2ee
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d1df7ad0a114c8a11094141079e2916c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed70f68191213dc4d797bf0a9a7c04a7
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_aa4eeb51e4482894d5500347a941a71f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 232, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_13f5594196c13a3c1120926b1dae5d15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 232, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9314caca6630e42ae25e2dd0ca17f4e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 232, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0ae6e1131fbdcbcc27618c0955620dbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 232, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_21e8cffff5520c95bd7e0a6db9af1d85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c605777dc616557595864ee0a0dca2a0
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ca746bdae046b750cb7040c4749e6b03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8bcfc8e34240d0d4ece23a1b4ff4e37
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_63fef9bef26ee305abbcb21bdbf49855(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ceb9af5b5895933fdaba0f3f45858e72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_db886c31be33688c507c255a9e1f3adc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 56, 56, 96], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_89840594af44b582bbabf7d1aa323f4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 40, 28, 28], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_de930accecaf552d6ec5b78b211c3639(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 40, 28, 28], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bfe837693d1d2aa7d5ffd2d68f73344b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 40, 28, 28], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_79eb4566f463d2f16a2858d4df490891(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 40, 28, 28], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_2a7be3742918d610e6bc1f114296e78b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 1, 2, None, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1e201967f8a0408cad97ddb330c28b7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a7be3742918d610e6bc1f114296e78b
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 1024, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d52e597c5dcd96217a4af4beb32eeb3a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 1, 2, None, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a9739ccdd57a31fc5bb1dbc7d296d6b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d52e597c5dcd96217a4af4beb32eeb3a
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 1024, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fcf77c276297769aec79576b8b51b1f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a7be3742918d610e6bc1f114296e78b
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 512, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e81309a72749e49a58e4c50f20949563(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d52e597c5dcd96217a4af4beb32eeb3a
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 512, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_628ee88ac7d3b8105a58a3b6ef55e268(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 97, 97], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_00124643d352da1c3ce83d2714240e78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 97, 97], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3d6e07f12d4a8ef75a8ecf118949cd38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 512, 8, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_92fab10a4052fdb48d8b0dc9a25b85e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 512, 8, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_997b3bd3badf232bf87ee0f19eb678b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 512, 8, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9ecc8afcdabdab1e6920f6d57a7598be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4842f9df2594d2180f180bfc7607814
    def get_inputs(self):
        return [
            paddle.uniform([3, 86, 3, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_431f1700698c8932dccfaba318462194(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2707d6d85b33ef86b3ceba242e1961
    def get_inputs(self):
        return [
            paddle.uniform([3, 86, 3, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_014250874501be692501d6db7a3d2dbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a35eadfbda236e342923f2b29a4c8b0f
    def get_inputs(self):
        return [
            paddle.uniform([3, 86, 3, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_047777555213c77e3dad7813f0e5a44a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7949ff795bd249c5d4af82cbc39ff51a
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 512, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5748be6e938e1081fbcf9e02a4226e8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e08bb01bfe322016e7a4cfc2ee9195a
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 512, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cf3fce9d89439b6750243299e405a456(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_01f837ac83a5e47d515fea9deef58155(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a62dbc811bdd7c8947c1be19ad90df30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_66d331d17ad788e44650a720c5ab4077(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8fe16338c581dbe06993ffeb0caf2057(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 512, 4, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_392c04f66d8e0b2cdde9530f972a05d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 512, 4, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_923d511861064cd47eb22b38426a139f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 512, 4, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0ac88a7ec80e0fe7efe049b871f1a589(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bb80b71efbe8cb6e8420d666d774e8b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_12f2a82b10c7b73cc1015f124ddec1cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 14, 14, 384], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e2028500f5843c339ea6fcff8dc7d632(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3024, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_16b6dae3c925f584c191a6f98eaa99c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3024, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bf1a87452b1dbdb567952afd316d5835(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5be1605a4558c34e97c36d03a8a086d6
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3024, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4fb1980d790ccf62f7a889251b3a8ef3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b7d6ee1bf2ea0f64f84e2f8952d8476c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4fb1980d790ccf62f7a889251b3a8ef3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_44a763a4cc3bcb30fe5640a95dd7e721(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_39aecdb3f8f1c7cc149996c3a3275190(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44a763a4cc3bcb30fe5640a95dd7e721
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_8bb41fc03c89066595b9fcacd1e22246(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ec0cc1116b315e8a87903c8ece56d0a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8bb41fc03c89066595b9fcacd1e22246
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_3c1a86fd830c5517015599288e59b858(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [3]
        input_2 = [4]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e58a009167f6ce7bbd15b2762dd00474(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c1a86fd830c5517015599288e59b858
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ac85c1f1acd7fb36793f63cbfae0ed6f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [4]
        input_2 = [5]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_49f4bfce18f61a99bf64a43cc9c49ee0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac85c1f1acd7fb36793f63cbfae0ed6f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_99a5cff246bf77a692a43bc9e8b2a40f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [5]
        input_2 = [6]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_841040c648c7d9a854423311bfd8e18a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99a5cff246bf77a692a43bc9e8b2a40f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_8303eadcb717736af7389eade08ab98e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [6]
        input_2 = [7]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5f8e271a4d00a68943fc652724eef989(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8303eadcb717736af7389eade08ab98e
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d48a30bac955a3638b06ce8d08ca3a60(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [7]
        input_2 = [8]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c50a4bbda875ece9cc2eb2536bd39083(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d48a30bac955a3638b06ce8d08ca3a60
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_073d25406a125d8032179e13d97b2b55(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [8]
        input_2 = [9]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cdadac12c893946ba398ffeec5a09e5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_073d25406a125d8032179e13d97b2b55
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_234466c494d103798cab43a4efd7adaa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [9]
        input_2 = [10]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f8733fa1f89608ca24bbf080787765ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_234466c494d103798cab43a4efd7adaa
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_7b57cd1704421f78fc07bf7519369281(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [10]
        input_2 = [11]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a233017dde9ea8d1b37be27a9723c982(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b57cd1704421f78fc07bf7519369281
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_bfb80f11e360e07ee069fda1c70656c1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [11]
        input_2 = [12]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b7169da3623c46cf9a5afe63666a2e3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfb80f11e360e07ee069fda1c70656c1
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ebbf9e0bf701438551bfc65f504c6431(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [12]
        input_2 = [13]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7b7e6c20381104afb3fdf9b20c5e7df2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ebbf9e0bf701438551bfc65f504c6431
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4398121e10ab8f599f0a6a5f99c131cc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [13]
        input_2 = [14]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_da856c9948f356a320a96959ee839287(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4398121e10ab8f599f0a6a5f99c131cc
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ba9d2eae54e23ed4d2f65fe03f45ba69(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [14]
        input_2 = [15]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c7cd05bfdea1c6235fdd2509e86ee070(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba9d2eae54e23ed4d2f65fe03f45ba69
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c1187114dc79cc98f2272cc230e2d278(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [15]
        input_2 = [16]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3596743ce44fb9297ea071166f540991(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1187114dc79cc98f2272cc230e2d278
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4118ef109871ddebe0b8d3690c9ad1b3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [16]
        input_2 = [17]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d6d5b01f17ccff8cad3a96513303cacd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4118ef109871ddebe0b8d3690c9ad1b3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
            paddle.to_tensor([17], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_1eb0ae1fab4dec61622dea2b399dfba9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [17]
        input_2 = [18]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9e09a20d1f076276d55386964e033aa7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1eb0ae1fab4dec61622dea2b399dfba9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([17], dtype='int64').reshape([1]),
            paddle.to_tensor([18], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d47e7f060352567e593dfce6bddb886a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [18]
        input_2 = [19]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_efc6badf8dcbc6f606094b81b4015636(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d47e7f060352567e593dfce6bddb886a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([18], dtype='int64').reshape([1]),
            paddle.to_tensor([19], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_0da479c8f2bd7f87a949e6dab4026964(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [19]
        input_2 = [20]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0ab8ed5c9e4b95d0fbcd47da1e500193(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0da479c8f2bd7f87a949e6dab4026964
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([19], dtype='int64').reshape([1]),
            paddle.to_tensor([20], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_dacb44700a95fe841bf21d3d12e321d2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [20]
        input_2 = [21]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8ae57aefd2f9b5ff33eabfe860a5ab1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dacb44700a95fe841bf21d3d12e321d2
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([20], dtype='int64').reshape([1]),
            paddle.to_tensor([21], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_1e6c2f7a0d546ead9704fee6d331e979(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [21]
        input_2 = [22]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_109e574da98609d972e94878dc89513b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e6c2f7a0d546ead9704fee6d331e979
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([21], dtype='int64').reshape([1]),
            paddle.to_tensor([22], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_5a9aef8c4883a2d9844154b2f3c5c01f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [22]
        input_2 = [23]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c4cb54466171424cacbd5faf714f6450(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a9aef8c4883a2d9844154b2f3c5c01f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([22], dtype='int64').reshape([1]),
            paddle.to_tensor([23], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_676e0d4c1f19c41bd86b721e9183bfa2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [23]
        input_2 = [24]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_da923fb50e772c7863380565ad0ab7ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_676e0d4c1f19c41bd86b721e9183bfa2
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([23], dtype='int64').reshape([1]),
            paddle.to_tensor([24], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4972088ebf9ece3939aac3951dbb94c2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [24]
        input_2 = [25]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_37a2f3b2aa534253b06c3621e941f70d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4972088ebf9ece3939aac3951dbb94c2
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([24], dtype='int64').reshape([1]),
            paddle.to_tensor([25], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_59c5726767198b8d9b55e4dbcf685765(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [25]
        input_2 = [26]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8871ff0f7a15667f1d591944ff0c6757(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59c5726767198b8d9b55e4dbcf685765
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([25], dtype='int64').reshape([1]),
            paddle.to_tensor([26], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_848cb015f4c8a1a00362329cf8b7c87a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [26]
        input_2 = [27]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_49cf9ec44b412a84c030f79bbd564ee7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_848cb015f4c8a1a00362329cf8b7c87a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([26], dtype='int64').reshape([1]),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4c63b22397c8c5514dfb2fc4b65a90f4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [27]
        input_2 = [28]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3e2768b41c2243dea6ac165ee2afee36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c63b22397c8c5514dfb2fc4b65a90f4
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
            paddle.to_tensor([28], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_680d9eb8a2b34dbdcfff9483f53fac3d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [28]
        input_2 = [29]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7cd7149827e6b2a08dc6c53601128276(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_680d9eb8a2b34dbdcfff9483f53fac3d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([28], dtype='int64').reshape([1]),
            paddle.to_tensor([29], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_64fe803ff9be28565528913b23c4048a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [29]
        input_2 = [30]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_88899cf0eacfc0f305a0db69040e1561(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64fe803ff9be28565528913b23c4048a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([29], dtype='int64').reshape([1]),
            paddle.to_tensor([30], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_fe351194e11fd4959808546b6517fa8b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [30]
        input_2 = [31]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_22dfb8a8aa48ef025afdc089d1688147(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe351194e11fd4959808546b6517fa8b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([30], dtype='int64').reshape([1]),
            paddle.to_tensor([31], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_79c5440400b2330b0ef831d322a58e22(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [31]
        input_2 = [32]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f2bc2d4cc847c188354d290fbc8e8156(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79c5440400b2330b0ef831d322a58e22
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([31], dtype='int64').reshape([1]),
            paddle.to_tensor([32], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_83fb2f528b4b2966a32cfe8ddd7f6317(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [32]
        input_2 = [33]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_62d7b06453e098f18b8a2525a0c4cf33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_83fb2f528b4b2966a32cfe8ddd7f6317
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([32], dtype='int64').reshape([1]),
            paddle.to_tensor([33], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_8a9f5040ce077977308ff9e15e3e098c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [33]
        input_2 = [34]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e0ff5657698a6abd2043c9c3faa58efb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a9f5040ce077977308ff9e15e3e098c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([33], dtype='int64').reshape([1]),
            paddle.to_tensor([34], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_8bf95e749ecf9e5c948ef9748f6e6a72(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [34]
        input_2 = [35]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f11b67b99af2aa4a40eb5b77c81f9ddb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8bf95e749ecf9e5c948ef9748f6e6a72
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([34], dtype='int64').reshape([1]),
            paddle.to_tensor([35], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f1f50b6742514188d2805fc966c16464(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [35]
        input_2 = [36]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c481c64b0774408f7b9014f20ca76470(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1f50b6742514188d2805fc966c16464
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([35], dtype='int64').reshape([1]),
            paddle.to_tensor([36], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_89c82d6be9e49b75d60ad86ccb7dc6b4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [36]
        input_2 = [37]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a43065643a57b9c933574a22f5d33dd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89c82d6be9e49b75d60ad86ccb7dc6b4
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([36], dtype='int64').reshape([1]),
            paddle.to_tensor([37], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_be1aaeb425bb5432529e73adbdc36c21(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [37]
        input_2 = [38]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_60004fe3d8fee4d40d5a82b02b5df061(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be1aaeb425bb5432529e73adbdc36c21
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([37], dtype='int64').reshape([1]),
            paddle.to_tensor([38], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_b58abc0cfa17c7f97d026ec515065024(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [38]
        input_2 = [39]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0fd895671eaaca682dd51663da16f675(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b58abc0cfa17c7f97d026ec515065024
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([38], dtype='int64').reshape([1]),
            paddle.to_tensor([39], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6370ff208add9ed59b78b059431d3821(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [39]
        input_2 = [40]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_230173c7dacb597a324cc086fa9d7837(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6370ff208add9ed59b78b059431d3821
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([39], dtype='int64').reshape([1]),
            paddle.to_tensor([40], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_e499cd2bb1a894a7d280d88213d4fff7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [40]
        input_2 = [41]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1b55d6ad522c0c58510d79d1b829b614(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e499cd2bb1a894a7d280d88213d4fff7
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([40], dtype='int64').reshape([1]),
            paddle.to_tensor([41], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_a949b9e1ca633bc71c30d1e1f2f6750c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [41]
        input_2 = [42]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8c005d3a075bdf20a018baf8a7c45a6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a949b9e1ca633bc71c30d1e1f2f6750c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([41], dtype='int64').reshape([1]),
            paddle.to_tensor([42], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_271fc14fa34ae1b2ea5f6113b17b2914(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [42]
        input_2 = [43]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_90f3968d1e3438022cba604cf92a7d7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_271fc14fa34ae1b2ea5f6113b17b2914
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([42], dtype='int64').reshape([1]),
            paddle.to_tensor([43], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_1ac6f6329447640a5f89f493f8729f0e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [43]
        input_2 = [44]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_419927edcf5e55d1d867e6bc4a12a8bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ac6f6329447640a5f89f493f8729f0e
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([43], dtype='int64').reshape([1]),
            paddle.to_tensor([44], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_a1e30665f438539fb1916075bb51ee20(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [44]
        input_2 = [45]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_067bc785fa1fc8c45322b04345f22331(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1e30665f438539fb1916075bb51ee20
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([44], dtype='int64').reshape([1]),
            paddle.to_tensor([45], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_026282c1930a794dc5216513160a5367(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [45]
        input_2 = [46]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7614c7323255d6e460bba9114a119fe4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_026282c1930a794dc5216513160a5367
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([45], dtype='int64').reshape([1]),
            paddle.to_tensor([46], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6657bd30dffd5524fab636db6e2d15c2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [46]
        input_2 = [47]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_48186cca26a097d93d06d9db1ce77272(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6657bd30dffd5524fab636db6e2d15c2
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([46], dtype='int64').reshape([1]),
            paddle.to_tensor([47], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_67d7d1498b0aeb35c1db42bb2785cab2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [47]
        input_2 = [48]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0a6ffb714f8ff1cc3430d3b5506da5ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67d7d1498b0aeb35c1db42bb2785cab2
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([47], dtype='int64').reshape([1]),
            paddle.to_tensor([48], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_a8b056707be5049a352d0f9e49e2403a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [48]
        input_2 = [49]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_659d60a3d43a24445675b86dce97f6f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8b056707be5049a352d0f9e49e2403a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([48], dtype='int64').reshape([1]),
            paddle.to_tensor([49], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_50e9c6c9ae634182395c66da05d8a853(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [49]
        input_2 = [50]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f7d99a5e10e7c53f9b24d67ff7ff0a66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50e9c6c9ae634182395c66da05d8a853
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([49], dtype='int64').reshape([1]),
            paddle.to_tensor([50], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_36b0409b1ee14426a73c005c1a385797(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [50]
        input_2 = [51]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bdb53619e2f75e9571a9fac3e4ec4d69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36b0409b1ee14426a73c005c1a385797
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([50], dtype='int64').reshape([1]),
            paddle.to_tensor([51], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_479889d75b7e606e61ee53bb5df4996e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [51]
        input_2 = [52]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_efffc72db53b5406dc3224421998315b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_479889d75b7e606e61ee53bb5df4996e
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([51], dtype='int64').reshape([1]),
            paddle.to_tensor([52], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_8b038c0c271b83ea7ce9c2201314d210(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [52]
        input_2 = [53]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5b9146f63aff69a8aea530de31db53b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b038c0c271b83ea7ce9c2201314d210
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([52], dtype='int64').reshape([1]),
            paddle.to_tensor([53], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c5e9cb794ae7ec23134ace8496c1ca32(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [53]
        input_2 = [54]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cb3cc7b8e57d80ce1d4ca4fa98becce1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5e9cb794ae7ec23134ace8496c1ca32
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([53], dtype='int64').reshape([1]),
            paddle.to_tensor([54], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_218b286430d50706b083dc5e8fec6b5c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [54]
        input_2 = [55]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_571c87abc4d7ce6872a11fd7ff8aec21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_218b286430d50706b083dc5e8fec6b5c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([54], dtype='int64').reshape([1]),
            paddle.to_tensor([55], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_20bfd3642c01e17317f5391d684c6650(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [55]
        input_2 = [56]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e63308c270bb8e0eaa273026c1f991f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20bfd3642c01e17317f5391d684c6650
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([55], dtype='int64').reshape([1]),
            paddle.to_tensor([56], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_9394636202eb59f1c039f747de09b147(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [56]
        input_2 = [57]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dd935cc33efde6ecc1aeaa49650a97f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9394636202eb59f1c039f747de09b147
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([56], dtype='int64').reshape([1]),
            paddle.to_tensor([57], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_b0f46647699ff5324f320c18383f747c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [57]
        input_2 = [58]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a1bd80ade639f2336758d2c8c218f633(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b0f46647699ff5324f320c18383f747c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([57], dtype='int64').reshape([1]),
            paddle.to_tensor([58], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_b05b06ffdcf5e2396cfe76a23d070db3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [58]
        input_2 = [59]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_200bdb8615d594221b50e35a74870db5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b05b06ffdcf5e2396cfe76a23d070db3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([58], dtype='int64').reshape([1]),
            paddle.to_tensor([59], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_3254b12a2abc8eda77e584a290a3a0d5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [59]
        input_2 = [60]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6b0f1b91b2a9552a1464fd02f6f04f2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3254b12a2abc8eda77e584a290a3a0d5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([59], dtype='int64').reshape([1]),
            paddle.to_tensor([60], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f62dd566efa3d81dea0b8f0dc066bbeb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [60]
        input_2 = [61]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e066819babb12efa05b79341659a303a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f62dd566efa3d81dea0b8f0dc066bbeb
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([60], dtype='int64').reshape([1]),
            paddle.to_tensor([61], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_3bda276b27bdc8d7ed8a7599ad96b580(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [61]
        input_2 = [62]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_da8bdb942c5f58ec590bb4b65d217d52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bda276b27bdc8d7ed8a7599ad96b580
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([61], dtype='int64').reshape([1]),
            paddle.to_tensor([62], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_7569d90c36c67e75081a1e7b3f1fc192(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [62]
        input_2 = [63]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6d0077086f8fcc696f2332bd174bb9b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7569d90c36c67e75081a1e7b3f1fc192
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([62], dtype='int64').reshape([1]),
            paddle.to_tensor([63], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_86a023f23d90be324ae992548a42d1f0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [63]
        input_2 = [64]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e2fd678648fa522ebd841a398490afda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86a023f23d90be324ae992548a42d1f0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([63], dtype='int64').reshape([1]),
            paddle.to_tensor([64], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_1a6ae7dace96681552c6180ee02279e3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [64]
        input_2 = [65]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c05793c7a99ca8141080e01a5e4e8862(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a6ae7dace96681552c6180ee02279e3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([64], dtype='int64').reshape([1]),
            paddle.to_tensor([65], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_da72dadcb16272c33c86473b25d5a211(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [65]
        input_2 = [66]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f09d6f6c6caedb31ece1e30ad2892cdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da72dadcb16272c33c86473b25d5a211
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([65], dtype='int64').reshape([1]),
            paddle.to_tensor([66], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_89d8e1e9c27dd10c21f46b2210e40abb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [66]
        input_2 = [67]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c83ba1728878cd79fee7bcfd6af8f00b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89d8e1e9c27dd10c21f46b2210e40abb
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([66], dtype='int64').reshape([1]),
            paddle.to_tensor([67], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_b44bebc64cf89446475d80b4d10b961a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [67]
        input_2 = [68]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_56e4a30bb3245e800588558936da2fc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b44bebc64cf89446475d80b4d10b961a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([67], dtype='int64').reshape([1]),
            paddle.to_tensor([68], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_a132cec5635473e80545522f5a473be8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [68]
        input_2 = [69]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eb1022b89798086a917ae167ae90a83d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a132cec5635473e80545522f5a473be8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([68], dtype='int64').reshape([1]),
            paddle.to_tensor([69], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_7f70c4c17059bb47bfbf448d59268b3e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [69]
        input_2 = [70]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_903c17070566bf1aaf966363abfb59ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f70c4c17059bb47bfbf448d59268b3e
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([69], dtype='int64').reshape([1]),
            paddle.to_tensor([70], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_dd2bf319b8ef0b4c96775dd85eb5d1ac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [70]
        input_2 = [71]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1e8d5fbeeec9c98263161294db4a7d1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd2bf319b8ef0b4c96775dd85eb5d1ac
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([70], dtype='int64').reshape([1]),
            paddle.to_tensor([71], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_9796cb80520817184f99b429ad446f66(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [71]
        input_2 = [72]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_12e69f75a4e4ae7f66ef9a775c3b7394(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9796cb80520817184f99b429ad446f66
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([71], dtype='int64').reshape([1]),
            paddle.to_tensor([72], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_435c1735c8fa551423e4a3046a4a6823(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [72]
        input_2 = [73]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b895bc9259cc15f675bd9bdeee9f3ea8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_435c1735c8fa551423e4a3046a4a6823
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([72], dtype='int64').reshape([1]),
            paddle.to_tensor([73], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ed0e16338e7a2464e4c6e179bcb2d6fa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [73]
        input_2 = [74]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_12108b2936c3cd51d74600ffab8cffc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed0e16338e7a2464e4c6e179bcb2d6fa
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([73], dtype='int64').reshape([1]),
            paddle.to_tensor([74], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_013b00db62dff8828eaa2de147496354(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [74]
        input_2 = [75]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f3376a72e74689436559983cf3d5d743(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_013b00db62dff8828eaa2de147496354
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([74], dtype='int64').reshape([1]),
            paddle.to_tensor([75], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4277ac4e29f7a154a50dcae38b8a393a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [75]
        input_2 = [76]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4460d861468c3912123c7e8128767157(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4277ac4e29f7a154a50dcae38b8a393a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([75], dtype='int64').reshape([1]),
            paddle.to_tensor([76], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_e20653f926f506decb17f09e24024481(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [76]
        input_2 = [77]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1782e2e3e8185e092b4f63117239aeca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e20653f926f506decb17f09e24024481
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([76], dtype='int64').reshape([1]),
            paddle.to_tensor([77], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_1878b5fd1e3b66e89d78b72d874cb02b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [77]
        input_2 = [78]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_48e98315f25c510c17363c191a981407(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1878b5fd1e3b66e89d78b72d874cb02b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([77], dtype='int64').reshape([1]),
            paddle.to_tensor([78], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_06e2df3d855ba4403a844c95d4959f53(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [78]
        input_2 = [79]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c2aed43c1ba6ae528b0f6bf48e7799a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06e2df3d855ba4403a844c95d4959f53
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([78], dtype='int64').reshape([1]),
            paddle.to_tensor([79], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f95a3d63515cac6a2e73c06c372e82f5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [79]
        input_2 = [80]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_055bc918202600f7de12b7062d5030b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f95a3d63515cac6a2e73c06c372e82f5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([79], dtype='int64').reshape([1]),
            paddle.to_tensor([80], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c08a1d087e0ac4c4ec7380dfdbeb7478(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [80]
        input_2 = [81]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_508898600879a663d8b6abdc041d7c00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c08a1d087e0ac4c4ec7380dfdbeb7478
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([80], dtype='int64').reshape([1]),
            paddle.to_tensor([81], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_2212332f8aa416ccd0fa44a5899c6424(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [81]
        input_2 = [82]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3b31b2c12e4455ae934bdf9c600d4e4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2212332f8aa416ccd0fa44a5899c6424
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([81], dtype='int64').reshape([1]),
            paddle.to_tensor([82], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_62cc2b8acca65ef4aaeabd1a5b260a19(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [82]
        input_2 = [83]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ac26bfe65889c0c7a118e60c523f63c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62cc2b8acca65ef4aaeabd1a5b260a19
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([82], dtype='int64').reshape([1]),
            paddle.to_tensor([83], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_7e09a2a369343c6c0e5c4802fd66550c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [83]
        input_2 = [84]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d3db7d18e98565a38f5e9e53653a8e75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e09a2a369343c6c0e5c4802fd66550c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([83], dtype='int64').reshape([1]),
            paddle.to_tensor([84], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_56c2d3c9b0a4d3dc93efdc0c8393d385(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [84]
        input_2 = [85]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bbc2fbf806e964cb2e3963329c8ff977(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56c2d3c9b0a4d3dc93efdc0c8393d385
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([84], dtype='int64').reshape([1]),
            paddle.to_tensor([85], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4e8ab90e950a2cc23cdd84783789fce9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [85]
        input_2 = [86]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9d90868f9af64f03b109a363c8b13389(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e8ab90e950a2cc23cdd84783789fce9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([85], dtype='int64').reshape([1]),
            paddle.to_tensor([86], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_5618684f36e88255e2e58bff43ccfc0e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [86]
        input_2 = [87]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_927696351e104e2f9a35a3f5518fd683(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5618684f36e88255e2e58bff43ccfc0e
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([86], dtype='int64').reshape([1]),
            paddle.to_tensor([87], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_5b8eb4bf8ffcc1806a44f037683c44eb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [87]
        input_2 = [88]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7ef88cb4551be372ff43669cb9693f4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b8eb4bf8ffcc1806a44f037683c44eb
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([87], dtype='int64').reshape([1]),
            paddle.to_tensor([88], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f0a7709fee812ecf0073bc0fc8f63fd9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [88]
        input_2 = [89]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6e79e7b3a2a9b825b971fd3fa9656efc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0a7709fee812ecf0073bc0fc8f63fd9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([88], dtype='int64').reshape([1]),
            paddle.to_tensor([89], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_eeb5847ecfa33de96d332ab713c73942(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [89]
        input_2 = [90]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_de30f219bfdf1af2296f897fb7fadae6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eeb5847ecfa33de96d332ab713c73942
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([89], dtype='int64').reshape([1]),
            paddle.to_tensor([90], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_482a69238ec7909ebb47b26ddf54062b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [90]
        input_2 = [91]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3934f76989fa2f09568e0505ff983002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_482a69238ec7909ebb47b26ddf54062b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([90], dtype='int64').reshape([1]),
            paddle.to_tensor([91], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_1249275437b6f66a5826680969dba9d2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [91]
        input_2 = [92]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_754d2eedd18e48b9ac5c340891178b37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1249275437b6f66a5826680969dba9d2
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([91], dtype='int64').reshape([1]),
            paddle.to_tensor([92], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_a77f45752d48b20f8990754d1c85a544(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [92]
        input_2 = [93]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_97aba4845364da4e9038210fb140fc06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a77f45752d48b20f8990754d1c85a544
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([92], dtype='int64').reshape([1]),
            paddle.to_tensor([93], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ac6fa09e099ce800ccb0fb48889b75a2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [93]
        input_2 = [94]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1231570572f62c9d911fcae011c85924(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac6fa09e099ce800ccb0fb48889b75a2
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([93], dtype='int64').reshape([1]),
            paddle.to_tensor([94], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_b3de144ea347e5aaaf98d1ee89200454(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [94]
        input_2 = [95]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_939d63c3bd97d504e3a9e3360309be5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3de144ea347e5aaaf98d1ee89200454
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([94], dtype='int64').reshape([1]),
            paddle.to_tensor([95], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_8852987433461b579932b15b8979c14f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [95]
        input_2 = [96]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_11939c4964ef38f83685fc7361d4e320(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8852987433461b579932b15b8979c14f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([95], dtype='int64').reshape([1]),
            paddle.to_tensor([96], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_325262c533e9145cb50e5aaa5b13e7a8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [96]
        input_2 = [97]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b59099f789a98dd5478a4b7c06912316(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_325262c533e9145cb50e5aaa5b13e7a8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([96], dtype='int64').reshape([1]),
            paddle.to_tensor([97], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_965b8aa76bcd69685fb7c749c41882e3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [97]
        input_2 = [98]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b8f7f88f7d067ee9e9579ef38bd5ea41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_965b8aa76bcd69685fb7c749c41882e3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([97], dtype='int64').reshape([1]),
            paddle.to_tensor([98], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_510100ade5503a52484660e29c685821(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [98]
        input_2 = [99]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_07dd572e0ee2115e635f72cf432da782(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_510100ade5503a52484660e29c685821
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([98], dtype='int64').reshape([1]),
            paddle.to_tensor([99], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_70872d17225e9e4b39c4e12067d4772a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [99]
        input_2 = [100]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_be57a1814e347efecb674ccc890f6a7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70872d17225e9e4b39c4e12067d4772a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([99], dtype='int64').reshape([1]),
            paddle.to_tensor([100], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_84e6584cf5ec61653b8e86be197e71c9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [100]
        input_2 = [101]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eff9bd4de11238765bc775556da2527c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84e6584cf5ec61653b8e86be197e71c9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([100], dtype='int64').reshape([1]),
            paddle.to_tensor([101], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_246f29ef981e9932c75f6e7051c806ad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [101]
        input_2 = [102]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f1976245c3734d90501e3fe23126d1f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_246f29ef981e9932c75f6e7051c806ad
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([101], dtype='int64').reshape([1]),
            paddle.to_tensor([102], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_717d6f4bf7f3fed29956ccdd1fa57d41(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [102]
        input_2 = [103]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_87a615fb0be4359e4c223b68b49f548f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_717d6f4bf7f3fed29956ccdd1fa57d41
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([102], dtype='int64').reshape([1]),
            paddle.to_tensor([103], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6afff3c9de91d268cdc76e48bb365716(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [103]
        input_2 = [104]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8147085a505a8b5c99c0db29a19eea37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6afff3c9de91d268cdc76e48bb365716
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([103], dtype='int64').reshape([1]),
            paddle.to_tensor([104], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4a752756fc7316180e9c4cedf2bbdd61(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [104]
        input_2 = [105]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_32f35bdd52fe35c2ea7b39ef66530051(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a752756fc7316180e9c4cedf2bbdd61
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([104], dtype='int64').reshape([1]),
            paddle.to_tensor([105], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_89f2f2056c2bfce4579008236f0b9312(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [105]
        input_2 = [106]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_60691eb9e7faefedd34813d863ba422f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89f2f2056c2bfce4579008236f0b9312
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([105], dtype='int64').reshape([1]),
            paddle.to_tensor([106], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_244e491f41dcf7398004dfc30a603486(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [106]
        input_2 = [107]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b52fc22fa88750a359a1900735a0c256(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_244e491f41dcf7398004dfc30a603486
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([106], dtype='int64').reshape([1]),
            paddle.to_tensor([107], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_13656d5eca901fc15db84f7624675156(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [107]
        input_2 = [108]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d039274ffacff9921f148aa907568108(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13656d5eca901fc15db84f7624675156
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([107], dtype='int64').reshape([1]),
            paddle.to_tensor([108], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_56b4851dbb3c61af5d0f0b9bf81e899a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [108]
        input_2 = [109]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e780697506c2137dcc75c4fe85b5e373(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56b4851dbb3c61af5d0f0b9bf81e899a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([108], dtype='int64').reshape([1]),
            paddle.to_tensor([109], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_e3ae72e1fbce9585e1012d0d574d1ed6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [109]
        input_2 = [110]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c3413f8e72a0083956bebdb2e79fceb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3ae72e1fbce9585e1012d0d574d1ed6
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([109], dtype='int64').reshape([1]),
            paddle.to_tensor([110], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_8c2461d104c67644c98540a4b95319e7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [110]
        input_2 = [111]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9b244f93b66da5befcd5dc3b5d301063(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c2461d104c67644c98540a4b95319e7
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([110], dtype='int64').reshape([1]),
            paddle.to_tensor([111], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_3e4e605fc90ee11aea1a570491fe2421(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [111]
        input_2 = [112]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b59c9b33491d13f8329188dd10590dff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e4e605fc90ee11aea1a570491fe2421
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([111], dtype='int64').reshape([1]),
            paddle.to_tensor([112], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_124171b4b65b313f02d63aee8ce4788e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [112]
        input_2 = [113]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8a34e51e6f3271bff5a6455e228ba65f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_124171b4b65b313f02d63aee8ce4788e
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([112], dtype='int64').reshape([1]),
            paddle.to_tensor([113], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_b0aae921947e1571aa4ce3a621922bd8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [113]
        input_2 = [114]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6648470d9a281116479243cb1499ca59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b0aae921947e1571aa4ce3a621922bd8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([113], dtype='int64').reshape([1]),
            paddle.to_tensor([114], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_8c0671c230e4d3541d810231063c702f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [114]
        input_2 = [115]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_38dca403ce587f6d9157cc2f266a5cec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c0671c230e4d3541d810231063c702f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([114], dtype='int64').reshape([1]),
            paddle.to_tensor([115], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_7672b6820a12e622c466e3743b939594(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [115]
        input_2 = [116]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6c365fedc189a092aa20ac1eab0a3587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7672b6820a12e622c466e3743b939594
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([115], dtype='int64').reshape([1]),
            paddle.to_tensor([116], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ad9c988a93e7f4b4e72cdecbc123db83(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [116]
        input_2 = [117]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_42140fd59bc67adf7f4ac0fee79c7fca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad9c988a93e7f4b4e72cdecbc123db83
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([116], dtype='int64').reshape([1]),
            paddle.to_tensor([117], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_a88962c2d16afacb97582d3a1fbbae20(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [117]
        input_2 = [118]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6b79f3b81ff9e4dc85e269cd6e15997c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a88962c2d16afacb97582d3a1fbbae20
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([117], dtype='int64').reshape([1]),
            paddle.to_tensor([118], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_9ac0f132eea76bcfc3e4e809255124c2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [118]
        input_2 = [119]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0f67df404339dd09ef6e166cf6bcab16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac0f132eea76bcfc3e4e809255124c2
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([118], dtype='int64').reshape([1]),
            paddle.to_tensor([119], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_8e614979639b106102cbde5dcdc2186e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [119]
        input_2 = [120]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2f1087f90ace0ba511fb82ebca83327f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e614979639b106102cbde5dcdc2186e
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([119], dtype='int64').reshape([1]),
            paddle.to_tensor([120], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_7ba342d74ac6fb5f57b93274884643e2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [120]
        input_2 = [121]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9323061de725098c838de7e8b0a57510(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ba342d74ac6fb5f57b93274884643e2
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([120], dtype='int64').reshape([1]),
            paddle.to_tensor([121], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_1e7fecd3dfe47d452692b776132e6892(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [121]
        input_2 = [122]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_afc18cfb05851904bb8250fb4fd1d668(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e7fecd3dfe47d452692b776132e6892
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([121], dtype='int64').reshape([1]),
            paddle.to_tensor([122], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4fefcb53c2ee4c896b27c05153e02f68(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [122]
        input_2 = [123]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1107669178b0a80490fea1520ecbd96a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4fefcb53c2ee4c896b27c05153e02f68
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([122], dtype='int64').reshape([1]),
            paddle.to_tensor([123], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_48470fe47f2e6236bb5ed25cb2ae9b46(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [123]
        input_2 = [124]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_65968203010e4f17db43fed83226bd12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48470fe47f2e6236bb5ed25cb2ae9b46
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([123], dtype='int64').reshape([1]),
            paddle.to_tensor([124], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_b273440b2eea42c8823d9f07fd4cd9d1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [124]
        input_2 = [125]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4eb5c0436fa106dfb6d594a48c7da496(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b273440b2eea42c8823d9f07fd4cd9d1
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([124], dtype='int64').reshape([1]),
            paddle.to_tensor([125], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4711f419535bab6e3a6c3370d98d60a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [125]
        input_2 = [126]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e9a2624f08f7b80a3c4eeafd07c0eccb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4711f419535bab6e3a6c3370d98d60a3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([125], dtype='int64').reshape([1]),
            paddle.to_tensor([126], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_a9a4e7701b497fd8071690524957d69c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [126]
        input_2 = [127]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2d16c594e3d2fc3a737834807817f0a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9a4e7701b497fd8071690524957d69c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([126], dtype='int64').reshape([1]),
            paddle.to_tensor([127], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_8c94f00cf7fc9d7b6ffacc83c80088fd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [127]
        input_2 = [128]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d73fd1b3dcac2d03e2a9ac964111d59a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c94f00cf7fc9d7b6ffacc83c80088fd
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([127], dtype='int64').reshape([1]),
            paddle.to_tensor([128], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_11cf3d47e032a57dbad370eac21fe0d0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [128]
        input_2 = [129]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cc89af60fe29eb225f2f31e444878f89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11cf3d47e032a57dbad370eac21fe0d0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([128], dtype='int64').reshape([1]),
            paddle.to_tensor([129], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f5e08001d13b131bcdf9f0fdbd1a9dae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [129]
        input_2 = [130]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0912ecd664852c33ca3b1e1a4608848d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5e08001d13b131bcdf9f0fdbd1a9dae
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([129], dtype='int64').reshape([1]),
            paddle.to_tensor([130], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_df0d86b15f744d9c4f1391eb6de8a64c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [130]
        input_2 = [131]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_28f357e4b84253351b77254bfeced369(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df0d86b15f744d9c4f1391eb6de8a64c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([130], dtype='int64').reshape([1]),
            paddle.to_tensor([131], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_50416f473de40b97af4ffdc9b1cac31d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [131]
        input_2 = [132]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_017c049cf81fdd52488547c4b901a974(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50416f473de40b97af4ffdc9b1cac31d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([131], dtype='int64').reshape([1]),
            paddle.to_tensor([132], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_be55366de927e198e2d29a45f3061181(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [132]
        input_2 = [133]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cb612ba334332e380e9535fe71b2b652(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be55366de927e198e2d29a45f3061181
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([132], dtype='int64').reshape([1]),
            paddle.to_tensor([133], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_383ed5251118053afd9a7697ab6d9707(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [133]
        input_2 = [134]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0e5e060b6b3086179b404bfc8f5e11a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_383ed5251118053afd9a7697ab6d9707
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([133], dtype='int64').reshape([1]),
            paddle.to_tensor([134], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4bbbaf2a772c1c2b09f30584b27d04be(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [134]
        input_2 = [135]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bcf2ef8183a182f237100073ad127311(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bbbaf2a772c1c2b09f30584b27d04be
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([134], dtype='int64').reshape([1]),
            paddle.to_tensor([135], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_b56de91bc4befd5d165cc7351bb9e8e9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [135]
        input_2 = [136]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a2cc9267d6ac8cd50ec6fe35f660f5d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b56de91bc4befd5d165cc7351bb9e8e9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([135], dtype='int64').reshape([1]),
            paddle.to_tensor([136], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_df52e3239546f7d56c7a0696353651ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [136]
        input_2 = [137]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_17da956de088e40318d5f686a45d47b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df52e3239546f7d56c7a0696353651ef
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([136], dtype='int64').reshape([1]),
            paddle.to_tensor([137], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_333c09493693f46ea4d2ea87147c7499(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [137]
        input_2 = [138]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_718e108e2e7edea342eaab048e56f398(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_333c09493693f46ea4d2ea87147c7499
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([137], dtype='int64').reshape([1]),
            paddle.to_tensor([138], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_61bfe6f87d4bcb1ea6551e3b51e36aeb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [138]
        input_2 = [139]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2e4b0f34c434b90c8988fa464746419a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61bfe6f87d4bcb1ea6551e3b51e36aeb
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([138], dtype='int64').reshape([1]),
            paddle.to_tensor([139], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_9130cdfda5e0f894ef5a5e54c251e5bb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [139]
        input_2 = [140]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a118ed5e350c0f43b054ab7444d6e423(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9130cdfda5e0f894ef5a5e54c251e5bb
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([139], dtype='int64').reshape([1]),
            paddle.to_tensor([140], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c7003858fb747afec7665c3a6839f9f5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [140]
        input_2 = [141]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2389954c19cd32b34accf595e2c9e09c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7003858fb747afec7665c3a6839f9f5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([140], dtype='int64').reshape([1]),
            paddle.to_tensor([141], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_fce37776e7fc394eebd7448840dfe8e3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [141]
        input_2 = [142]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_89238cc74e104a0584ecd4c963c3159f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fce37776e7fc394eebd7448840dfe8e3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([141], dtype='int64').reshape([1]),
            paddle.to_tensor([142], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_1d28d524c5940e5bbbf11c09fa31313c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [142]
        input_2 = [143]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_311c3da4efb40206b2553668f02e8799(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d28d524c5940e5bbbf11c09fa31313c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([142], dtype='int64').reshape([1]),
            paddle.to_tensor([143], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_8683641b5160a5fcc2b5c5bc4a861f6d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [143]
        input_2 = [144]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_641127db0475efb17457a8a81d7d1c6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8683641b5160a5fcc2b5c5bc4a861f6d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([143], dtype='int64').reshape([1]),
            paddle.to_tensor([144], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_019906f0fa80d1ab6efa6cfa4d6e0474(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [144]
        input_2 = [145]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d51045cfb60104a7ba7c445f511b9b18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_019906f0fa80d1ab6efa6cfa4d6e0474
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([144], dtype='int64').reshape([1]),
            paddle.to_tensor([145], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ebecb54042dc3371e37031c527dec7be(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [145]
        input_2 = [146]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_38c4bc8fd2d91d184f7f1d2c1ecea9e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ebecb54042dc3371e37031c527dec7be
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([145], dtype='int64').reshape([1]),
            paddle.to_tensor([146], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4f9de924e81d92fd692159fa60645e7e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [146]
        input_2 = [147]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_db45c0c2b6c3c08ccced6fd5d2fb8798(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f9de924e81d92fd692159fa60645e7e
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([146], dtype='int64').reshape([1]),
            paddle.to_tensor([147], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f3a8c6265dc38ec71cef8b7a95057dcc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [147]
        input_2 = [148]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9bc6bea1421d1815fddc6d9eb2d36d0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3a8c6265dc38ec71cef8b7a95057dcc
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([147], dtype='int64').reshape([1]),
            paddle.to_tensor([148], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_7f092b7606b17384b09f7786fb56c5d1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [148]
        input_2 = [149]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e0d1c925c5ab180fd9c6290456d905b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f092b7606b17384b09f7786fb56c5d1
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([148], dtype='int64').reshape([1]),
            paddle.to_tensor([149], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_7597706774dfffe102b12fa6f5926453(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [149]
        input_2 = [150]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_450733d85783d789554d079b1f8da4f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7597706774dfffe102b12fa6f5926453
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([149], dtype='int64').reshape([1]),
            paddle.to_tensor([150], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_7ca9fd5ac6b199d63f4892ef77b02bf6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [150]
        input_2 = [151]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f87147a5d65ac3cecf8a2887aa493242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca9fd5ac6b199d63f4892ef77b02bf6
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([150], dtype='int64').reshape([1]),
            paddle.to_tensor([151], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_8eb55f0af8507060a82126ebbc974144(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [151]
        input_2 = [152]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4cbc377f8ce8000eada9a236769c1074(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8eb55f0af8507060a82126ebbc974144
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([151], dtype='int64').reshape([1]),
            paddle.to_tensor([152], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_43228ecf4b359a95411f0547aadbd9d0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [152]
        input_2 = [153]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_09f0a42c5398fca85b626d80a028e0d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43228ecf4b359a95411f0547aadbd9d0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([152], dtype='int64').reshape([1]),
            paddle.to_tensor([153], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_7cd604c4aab3585f94fdad48961953eb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [153]
        input_2 = [154]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eddee3182a3094e13af59c90330b29ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cd604c4aab3585f94fdad48961953eb
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([153], dtype='int64').reshape([1]),
            paddle.to_tensor([154], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_13783f985ce0863243e194f52b9f86cd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [154]
        input_2 = [155]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_119ab894dedd49de152b53fd80ea6add(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13783f985ce0863243e194f52b9f86cd
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([154], dtype='int64').reshape([1]),
            paddle.to_tensor([155], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ed43e525afb19050d31f0acf17ec1482(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [155]
        input_2 = [156]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_74aa42d365dac9c8ec3add58cff188a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed43e525afb19050d31f0acf17ec1482
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([155], dtype='int64').reshape([1]),
            paddle.to_tensor([156], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_5cb0b9632b9e8ad6b83402abd2a6f856(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [156]
        input_2 = [157]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4d747cd6bc66793c6c413e6ffc263224(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5cb0b9632b9e8ad6b83402abd2a6f856
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([156], dtype='int64').reshape([1]),
            paddle.to_tensor([157], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_2e4e86e6d231f75c5795c61e4104444a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [157]
        input_2 = [158]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_63afa7ae508196525e0272396d7a2d0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2e4e86e6d231f75c5795c61e4104444a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([157], dtype='int64').reshape([1]),
            paddle.to_tensor([158], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_e5fabdbff38b1a1852911d2798e18a2d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [158]
        input_2 = [159]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3c4187d02d9c13f8021312c3f84bf59a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5fabdbff38b1a1852911d2798e18a2d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([158], dtype='int64').reshape([1]),
            paddle.to_tensor([159], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_9d6fd50235c1312a089e8376609b080b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [159]
        input_2 = [160]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_658643f1c8a1f1bacb04b2d3e42579c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d6fd50235c1312a089e8376609b080b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([159], dtype='int64').reshape([1]),
            paddle.to_tensor([160], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_8aa8592317595b24a0323d08772d37e8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [160]
        input_2 = [161]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1b75b0a1dd61f27fe4b617bdece24cad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8aa8592317595b24a0323d08772d37e8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([160], dtype='int64').reshape([1]),
            paddle.to_tensor([161], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c83ec3df43a8e99d568a44f51e96f3ec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [161]
        input_2 = [162]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a62dafb2eef27a114b49fc035c9b2cd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c83ec3df43a8e99d568a44f51e96f3ec
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([161], dtype='int64').reshape([1]),
            paddle.to_tensor([162], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ece7a946084c5f311a528a3aa40d702f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [162]
        input_2 = [163]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_24253a8e274afd3a6f379a0f5979a299(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ece7a946084c5f311a528a3aa40d702f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([162], dtype='int64').reshape([1]),
            paddle.to_tensor([163], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_75d3f83f2937302457c1b5f7a00b45ca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [163]
        input_2 = [164]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_897a4dcc2ad46dbfe5a9b2e836e42392(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75d3f83f2937302457c1b5f7a00b45ca
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([163], dtype='int64').reshape([1]),
            paddle.to_tensor([164], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_8290d307be7fd74c7b20367b422f5b57(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [164]
        input_2 = [165]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8dba44b3e7549a20af58f453e20ac65d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8290d307be7fd74c7b20367b422f5b57
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([164], dtype='int64').reshape([1]),
            paddle.to_tensor([165], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d17053891c1c5848a83705dcf278853d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [165]
        input_2 = [166]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c03030f85ce63fcfb345e132bda23269(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d17053891c1c5848a83705dcf278853d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([165], dtype='int64').reshape([1]),
            paddle.to_tensor([166], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_e414a7aabdc21fdedad31c7228790880(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [166]
        input_2 = [167]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b946c50dae9b041d47eb83f4349a3b6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e414a7aabdc21fdedad31c7228790880
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([166], dtype='int64').reshape([1]),
            paddle.to_tensor([167], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_759c8c48fa45241d7780cf4385c7e4a4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [167]
        input_2 = [168]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a075dd8fca965587d8fe371fde8d5937(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_759c8c48fa45241d7780cf4385c7e4a4
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([167], dtype='int64').reshape([1]),
            paddle.to_tensor([168], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_fef3610bb76c2605ed466bbd163e9ce0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [168]
        input_2 = [169]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1c864cf7490cb5a97b51ddeb3de4ad8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fef3610bb76c2605ed466bbd163e9ce0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([168], dtype='int64').reshape([1]),
            paddle.to_tensor([169], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c22b166131259e6781376748c96fa06c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [169]
        input_2 = [170]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_75a40431ca4b2a48b5f7e7a0522610a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c22b166131259e6781376748c96fa06c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([169], dtype='int64').reshape([1]),
            paddle.to_tensor([170], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_88c31694c4eb2f2fad0c0f391ca85a3e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [170]
        input_2 = [171]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_40acd05518b240634274243a1359542c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88c31694c4eb2f2fad0c0f391ca85a3e
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([170], dtype='int64').reshape([1]),
            paddle.to_tensor([171], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_12a7521fe01167ea3c4b806924780343(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [171]
        input_2 = [172]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cb4f247256f9c5160eb668f1c4c37906(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12a7521fe01167ea3c4b806924780343
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([171], dtype='int64').reshape([1]),
            paddle.to_tensor([172], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_1ca9e7ab3d69e9ad0abce94c13e32cf8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [172]
        input_2 = [173]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2ed5155e6d2b95f8d815016cc047bd60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ca9e7ab3d69e9ad0abce94c13e32cf8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([172], dtype='int64').reshape([1]),
            paddle.to_tensor([173], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_7c81cdda0a12cb9d91cfe3d5b9ae92d5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [173]
        input_2 = [174]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_df40a3696d583c59ff7eb0a3e52b29c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c81cdda0a12cb9d91cfe3d5b9ae92d5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([173], dtype='int64').reshape([1]),
            paddle.to_tensor([174], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_83e4ce801b4e2ead9a3ec0c32d2c3a11(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [174]
        input_2 = [175]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_897d86cb8a7cced148aea86c29d84bf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_83e4ce801b4e2ead9a3ec0c32d2c3a11
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([174], dtype='int64').reshape([1]),
            paddle.to_tensor([175], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_fdbd84a9fd9fe5c8fe4134796dae3775(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [175]
        input_2 = [176]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_46a1e8bccbb08b803f8289f938c309d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fdbd84a9fd9fe5c8fe4134796dae3775
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([175], dtype='int64').reshape([1]),
            paddle.to_tensor([176], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_39670881cd3a4c3fad724c20e09e3f53(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [176]
        input_2 = [177]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_66e193293f0ee34cba064087e34d1d4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39670881cd3a4c3fad724c20e09e3f53
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([176], dtype='int64').reshape([1]),
            paddle.to_tensor([177], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_3d86185091b509a77990b7287094821d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [177]
        input_2 = [178]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cc5c452673867e367c7de42fcd9ed92c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d86185091b509a77990b7287094821d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([177], dtype='int64').reshape([1]),
            paddle.to_tensor([178], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_02712159ae6c8ed6ffe26651ef996591(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [178]
        input_2 = [179]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8006d3bf42b6b22dc78dece5938351a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02712159ae6c8ed6ffe26651ef996591
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([178], dtype='int64').reshape([1]),
            paddle.to_tensor([179], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_484970b14510aa5a6b6e85c153eb37b8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [179]
        input_2 = [180]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_259a63098bbcaf995801961b1d37f6db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_484970b14510aa5a6b6e85c153eb37b8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([179], dtype='int64').reshape([1]),
            paddle.to_tensor([180], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_3f70d356f5545e47c9bfc2c3ffef70a8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [180]
        input_2 = [181]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9bf6138bdf56e292ac116f848928a9b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f70d356f5545e47c9bfc2c3ffef70a8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([180], dtype='int64').reshape([1]),
            paddle.to_tensor([181], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_9eb12708eaac9db87347941e34dd4640(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [181]
        input_2 = [182]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_527289a8093204fbefa798cfc7b3ac73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9eb12708eaac9db87347941e34dd4640
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([181], dtype='int64').reshape([1]),
            paddle.to_tensor([182], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6395234bd6365a3d95fd8c351bd5b3e9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [182]
        input_2 = [183]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e12b67aa2ffebe0ce0687d42e4fa1ec7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6395234bd6365a3d95fd8c351bd5b3e9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([182], dtype='int64').reshape([1]),
            paddle.to_tensor([183], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6e4ad40e5dea928b65ad063c126b98d6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [183]
        input_2 = [184]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d1695efa084af8f6ba6508ff9103c6f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e4ad40e5dea928b65ad063c126b98d6
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([183], dtype='int64').reshape([1]),
            paddle.to_tensor([184], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_fcba45baa35ecaed996eb6356df86e2d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [184]
        input_2 = [185]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2607ae52e6a8ab566556db90518342ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fcba45baa35ecaed996eb6356df86e2d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([184], dtype='int64').reshape([1]),
            paddle.to_tensor([185], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_00c3b7eea295752efebd92dc47f8b85b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [185]
        input_2 = [186]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2fbaf6e8541bb479d78e7d4b20116734(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00c3b7eea295752efebd92dc47f8b85b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([185], dtype='int64').reshape([1]),
            paddle.to_tensor([186], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6b0778ed19fae1ceed49ce7c875f7650(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [186]
        input_2 = [187]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_78a0d2912e5cec363342708053a7ab09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b0778ed19fae1ceed49ce7c875f7650
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([186], dtype='int64').reshape([1]),
            paddle.to_tensor([187], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_7c63a3ed4de6395baee8f43f2742b41a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [187]
        input_2 = [188]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e4fe2605e9fef4f59a6c3c684bb83b00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c63a3ed4de6395baee8f43f2742b41a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([187], dtype='int64').reshape([1]),
            paddle.to_tensor([188], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_3bfce9bb23072a9e2035cf28f900d1ea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [188]
        input_2 = [189]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_49af09714e5e235c4c9c42eae08a94d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bfce9bb23072a9e2035cf28f900d1ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([188], dtype='int64').reshape([1]),
            paddle.to_tensor([189], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d9960897c44cba11fe8257116d51de4a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [189]
        input_2 = [190]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7c78195c02118efa70fd0b7c4e3ee94e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9960897c44cba11fe8257116d51de4a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([189], dtype='int64').reshape([1]),
            paddle.to_tensor([190], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_16da50b6ca8fb4ef0d7662b88d9e6e4b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [190]
        input_2 = [191]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_438b5960c7f8e0fd8690c0757afac927(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16da50b6ca8fb4ef0d7662b88d9e6e4b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([190], dtype='int64').reshape([1]),
            paddle.to_tensor([191], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_8ece65a5279503787e8336a75da5c453(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [191]
        input_2 = [192]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_50bd9d997f0a152e2dd70767b25f899b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ece65a5279503787e8336a75da5c453
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([191], dtype='int64').reshape([1]),
            paddle.to_tensor([192], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d179bef28e3a068a2704ef0be21bf1d3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [192]
        input_2 = [193]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f20e08aa4b7f1053709a53c1839dfe2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d179bef28e3a068a2704ef0be21bf1d3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([192], dtype='int64').reshape([1]),
            paddle.to_tensor([193], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c56411777c22b658ef58ade42223bf96(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [193]
        input_2 = [194]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_996684cb41468cf944c5679c86113cfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c56411777c22b658ef58ade42223bf96
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([193], dtype='int64').reshape([1]),
            paddle.to_tensor([194], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_3f044ef383f2335a19c0eeec75caea9f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [194]
        input_2 = [195]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6ef485eb17795090ded6148641cf7f3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f044ef383f2335a19c0eeec75caea9f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([194], dtype='int64').reshape([1]),
            paddle.to_tensor([195], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c137d355525356b71794abfdc514c84c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [195]
        input_2 = [196]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5160ead729579804e254637ac2fd81f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c137d355525356b71794abfdc514c84c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([195], dtype='int64').reshape([1]),
            paddle.to_tensor([196], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ddd68c4673164cda3ea7dd09f0008629(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 8, None, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_479512c8d1700981977e497960d3eb97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ddd68c4673164cda3ea7dd09f0008629
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 8, 160, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6cb5625ac551fc45d8f35c17e21c2bc0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 8, None, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fdf2aab5e4e34440b8234dc78d9bde1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6cb5625ac551fc45d8f35c17e21c2bc0
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 8, 160, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_b306c0f3e6e955ed468d2cae2e394bb0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 8, None, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d21bf8a839e1487a6df61e98781e10b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b306c0f3e6e955ed468d2cae2e394bb0
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 8, 160, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_929cbe60d5d611b893f372f1cfd8bf8a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b9dd834d350d5dc996a5f9537d84d3c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_929cbe60d5d611b893f372f1cfd8bf8a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_468a26531d21b45e770c9805faac3308(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c3c519b30ccec94a8cd2cf8b1ea6d20a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_468a26531d21b45e770c9805faac3308
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c5cd0a70746ab9ca94962a9d5bbc99b7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bf69aba17ac54debcc11bae10c2a8f9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5cd0a70746ab9ca94962a9d5bbc99b7
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4755495bcdd84d346928d005f1d859bd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [3]
        input_2 = [4]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_127bc0115a8de2b83c5790b66fa9712e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4755495bcdd84d346928d005f1d859bd
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ea6dec648543127c8be6f5d1f1db5ea5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [4]
        input_2 = [5]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9c9ea3f0a7e595d1b4e739badc115285(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea6dec648543127c8be6f5d1f1db5ea5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_b72f895ddfe49b771310ecf55afa56f3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [5]
        input_2 = [6]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f65c84061bff960f766c89551b27f06b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b72f895ddfe49b771310ecf55afa56f3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c51a5ca0949df8178295f07f9ca17451(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [6]
        input_2 = [7]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3da7654c64afb5aabd299ec76a74b604(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51a5ca0949df8178295f07f9ca17451
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f3bac0d64af217bea0839e677c6fe7d7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [7]
        input_2 = [8]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_21760b8a33adb1acb5d2ae5ac800f295(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3bac0d64af217bea0839e677c6fe7d7
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_971c0e4e1d770e3296c578516a8dde3c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [8]
        input_2 = [9]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6d4112398fab8c63c776d4531ee37bad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_971c0e4e1d770e3296c578516a8dde3c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_eb7dd9d0bee13a9508e0eb67dad73198(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [9]
        input_2 = [10]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a0f684a982643979a5dab6c271dd7b83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb7dd9d0bee13a9508e0eb67dad73198
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_003c2610c54a438a5df9673f77449011(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [10]
        input_2 = [11]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_95093141c064744c3ee2237951267aa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_003c2610c54a438a5df9673f77449011
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_266c5d33a801f77a0cf55f93c95d6c34(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [11]
        input_2 = [12]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_24121301177f7ac26f95f514502d3e47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_266c5d33a801f77a0cf55f93c95d6c34
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_fceee60ee7d31d0137ab01e0ed05443d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [12]
        input_2 = [13]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_54d87351dc1d40887d4eec5b500aaaee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fceee60ee7d31d0137ab01e0ed05443d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_78ae70e6a62147157f8d5ba58c62f00c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [13]
        input_2 = [14]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3d8cf2df96e8d843a49f669b7926efa4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78ae70e6a62147157f8d5ba58c62f00c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_2efd99d732bca1b9de0554355a865d82(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [14]
        input_2 = [15]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_57f9631a61451c52bc2f3ffd7b261622(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2efd99d732bca1b9de0554355a865d82
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6a2279e22021bfafa97ea0869a91243e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [15]
        input_2 = [16]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_96532494a0265dfb926fe16c99d2392f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a2279e22021bfafa97ea0869a91243e
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d46781a5b98e6a37f7bb7ab5440c390f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [16]
        input_2 = [17]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c8bd89ae6dcd0efe610cfd7e6db43cb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d46781a5b98e6a37f7bb7ab5440c390f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
            paddle.to_tensor([17], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_744a2d925a56f11c548c6059e393da69(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [17]
        input_2 = [18]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f956e3b3792533781a2b49693e563ad0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_744a2d925a56f11c548c6059e393da69
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([17], dtype='int64').reshape([1]),
            paddle.to_tensor([18], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_eca4f076c8655e339cee7854ea16bc74(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [18]
        input_2 = [19]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2f31825b564be4c4e339e73f324ae8cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca4f076c8655e339cee7854ea16bc74
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([18], dtype='int64').reshape([1]),
            paddle.to_tensor([19], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_b3a733ed8b4d9f736b007408b2dd123d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [19]
        input_2 = [20]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0847676eb6c9aeb5793977f7cfd22c59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3a733ed8b4d9f736b007408b2dd123d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([19], dtype='int64').reshape([1]),
            paddle.to_tensor([20], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_8b7576b86906a771e5084fe9d2e894b6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [20]
        input_2 = [21]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6f306168a2887ba4e9af4f16dcbfd557(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b7576b86906a771e5084fe9d2e894b6
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([20], dtype='int64').reshape([1]),
            paddle.to_tensor([21], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_9aca934bed71a8d4c5c806bd071da508(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [21]
        input_2 = [22]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b190a54b7110bc1b86028138e59ce334(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9aca934bed71a8d4c5c806bd071da508
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([21], dtype='int64').reshape([1]),
            paddle.to_tensor([22], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_0371360dc2271c300801d9432b6eaede(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [22]
        input_2 = [23]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_381742889f8cd93ce35dd1eb0cff1cb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0371360dc2271c300801d9432b6eaede
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([22], dtype='int64').reshape([1]),
            paddle.to_tensor([23], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_7a23a6eed77f5c253daffe200528b516(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [23]
        input_2 = [24]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_63c1c146c044ec9f098acce96f072365(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a23a6eed77f5c253daffe200528b516
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([23], dtype='int64').reshape([1]),
            paddle.to_tensor([24], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c5b565f9c9f148d7725714a0ba3ecffe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [24]
        input_2 = [25]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8f8f2612661a4411d54241fd9f08b434(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5b565f9c9f148d7725714a0ba3ecffe
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([24], dtype='int64').reshape([1]),
            paddle.to_tensor([25], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_9480e64ed6cb5a48491d1ee5457ddcce(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [25]
        input_2 = [26]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_60a47fbbbb8fa47a7ba5b787c3b86f8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9480e64ed6cb5a48491d1ee5457ddcce
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([25], dtype='int64').reshape([1]),
            paddle.to_tensor([26], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_9631a99a2e8568233e61003409459a92(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [26]
        input_2 = [27]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9ca5480c376ef6e643b825afca9f8f0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9631a99a2e8568233e61003409459a92
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([26], dtype='int64').reshape([1]),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_80552ac9081599651d4aef221f0a2be3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [27]
        input_2 = [28]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b82b124aa2617cc2485bb935c2dfc399(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80552ac9081599651d4aef221f0a2be3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
            paddle.to_tensor([28], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_2c1baa4bcd40f5a4a396e5adb9de6891(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [28]
        input_2 = [29]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c8284b5d47c4bb95ac745eb2e9d79e31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c1baa4bcd40f5a4a396e5adb9de6891
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([28], dtype='int64').reshape([1]),
            paddle.to_tensor([29], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_03d3639c7f82ca396d70b9973c68d7f5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [29]
        input_2 = [30]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_39106396f7f567bfde10d29d204b7f16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03d3639c7f82ca396d70b9973c68d7f5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([29], dtype='int64').reshape([1]),
            paddle.to_tensor([30], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ae2a3d5c80c664b6dce81fc91b580257(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [30]
        input_2 = [31]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c865d9557357af47262d858305a41c95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae2a3d5c80c664b6dce81fc91b580257
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([30], dtype='int64').reshape([1]),
            paddle.to_tensor([31], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c86c5a45d91cbc4b3e02f5ddc01b851d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [31]
        input_2 = [32]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a410a69cb96e6ee11c6752b93bd3b76f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c86c5a45d91cbc4b3e02f5ddc01b851d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([31], dtype='int64').reshape([1]),
            paddle.to_tensor([32], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6362254caa66cafdb2852ab6094fc8c4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [32]
        input_2 = [33]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_af1b4f52762f8a406ffde292d26b524c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6362254caa66cafdb2852ab6094fc8c4
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([32], dtype='int64').reshape([1]),
            paddle.to_tensor([33], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_74ca898d3ee3be33197e0eae0f6af273(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [33]
        input_2 = [34]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_12962ff53fb9e42861a71f09f1c20feb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74ca898d3ee3be33197e0eae0f6af273
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([33], dtype='int64').reshape([1]),
            paddle.to_tensor([34], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_a730a7f9af03b360e70c16488a0661a4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [34]
        input_2 = [35]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c60b31d3d565688129c07abee0bb00d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a730a7f9af03b360e70c16488a0661a4
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([34], dtype='int64').reshape([1]),
            paddle.to_tensor([35], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_323f53cba4e9ea8585c13fae5637e026(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [35]
        input_2 = [36]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5fe8fd7e731e65958d4a4a1afe3f1699(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_323f53cba4e9ea8585c13fae5637e026
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([35], dtype='int64').reshape([1]),
            paddle.to_tensor([36], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_db779c0cb62fcc613512ab41a56381d9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [36]
        input_2 = [37]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dae6de17f3e18996c0d94fb273d8e689(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db779c0cb62fcc613512ab41a56381d9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([36], dtype='int64').reshape([1]),
            paddle.to_tensor([37], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_34a5605370c35f12a1eebbe2747d27dc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [37]
        input_2 = [38]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5afda3e54b8b7fd996caa1609fca7c35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34a5605370c35f12a1eebbe2747d27dc
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([37], dtype='int64').reshape([1]),
            paddle.to_tensor([38], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_b3b488f4c5b75a95a6e439a35b449999(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [38]
        input_2 = [39]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b45b77e6572f4cc3acaba25b125d4137(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3b488f4c5b75a95a6e439a35b449999
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([38], dtype='int64').reshape([1]),
            paddle.to_tensor([39], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_1abcc0105161aaca3008e7f20d041488(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [39]
        input_2 = [40]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1e49ab236d55bdf00a3e9e3495824068(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1abcc0105161aaca3008e7f20d041488
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([39], dtype='int64').reshape([1]),
            paddle.to_tensor([40], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ec628c6be1da6d00849a5010f289a427(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [40]
        input_2 = [41]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dc32030bad17e26f99ee133e6a7352d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec628c6be1da6d00849a5010f289a427
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([40], dtype='int64').reshape([1]),
            paddle.to_tensor([41], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_75e56295cd40a0b18a6e3edb753ebb8e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [41]
        input_2 = [42]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fbff56bcd6c4bc94a5584e2adf3aabe9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75e56295cd40a0b18a6e3edb753ebb8e
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([41], dtype='int64').reshape([1]),
            paddle.to_tensor([42], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_a361bc4d73c7c31d3dac5e1b628b7693(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [42]
        input_2 = [43]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bf7c0c9bbc65d18e80ec715185d63832(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a361bc4d73c7c31d3dac5e1b628b7693
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([42], dtype='int64').reshape([1]),
            paddle.to_tensor([43], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_dd02a80cf0b5effa329b6190e054b783(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [43]
        input_2 = [44]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fdaef2e8f6a27d210e3df2db28a7291c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd02a80cf0b5effa329b6190e054b783
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([43], dtype='int64').reshape([1]),
            paddle.to_tensor([44], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_332916638df22fd42463bc34880d9193(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [44]
        input_2 = [45]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2f777f25d46c4c5c7bb58af41141c7cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_332916638df22fd42463bc34880d9193
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([44], dtype='int64').reshape([1]),
            paddle.to_tensor([45], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ebe47257c6bde9065e2ec4facca6c999(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [45]
        input_2 = [46]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d3e18e425705146b495b044bd2b78b6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ebe47257c6bde9065e2ec4facca6c999
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([45], dtype='int64').reshape([1]),
            paddle.to_tensor([46], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_1a65f94bbfbe1fb2390dbe430f7ccdeb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [46]
        input_2 = [47]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_347b5826d350158b446a7010b4332688(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a65f94bbfbe1fb2390dbe430f7ccdeb
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([46], dtype='int64').reshape([1]),
            paddle.to_tensor([47], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_e20cfd45136207876a32bb6126c39dd2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [47]
        input_2 = [48]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b3c5dcb172a83d3b78698ed4b48ffc8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e20cfd45136207876a32bb6126c39dd2
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([47], dtype='int64').reshape([1]),
            paddle.to_tensor([48], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_8c0f5f100b2da00697d61f2c0be436a0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [48]
        input_2 = [49]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7e3a7f44c65e675f9d9a6a767df73d37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c0f5f100b2da00697d61f2c0be436a0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([48], dtype='int64').reshape([1]),
            paddle.to_tensor([49], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5974fbaeca1630bf015fb4cadcabe6d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_accf37a0f5a564156a0890c1a01c8d1e
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8432d7b69537f7c22d95aa62cdb56616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ba9e3ecc7aefd07a12a926630167a2b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_77f5617fea127f7a0450ff254a5be5c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41aa0f99178c08b47f1ea24271aae0de
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c376f72a2f4cccdeff1784475da68605(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8991cc386dfe6077ba1e45fea25759c9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_349fb530a3c8d2c5309c12384632e5d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86e923be65b1a3560154e7d03e4a44ed
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c64c79355abf695eb3f7e532bfad464c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f6f8f5fdfcf4fd8fc4d6731a4502567
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_af73e3945e0749640518426c75329984(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c931bd587049811f872a1f5e314125
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5abdac38b0e5be00f14c193a16228bc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8f675979b95c010b78ef4e55ad8ec33
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_aecfc14f0712bb2d13a2ee41dbf05856(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d079da2c40e04afcee7253847677c69c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a84c7b708bc1c7f4d8eafa352a6be0e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50d446c5fdf3f32ad804b5fda30dcd16
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_055c59807c3eba6c032bbe64bde6ce39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_855b9b2db82e33602b1fbbd78e5ffe02
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b7bcaa21aa8f833b6394f49eed9ab74a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f80b9a16ea71fa6c0f99cc34936b0bae
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_37c2cbe072d77ebab07fe8589894fdee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9960c973a9f0a71a72d24c83dab22c34
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_25fca5662637e66dc450f230111296c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07d8806293c26737ea42be3fe3f97f3a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3258830271581514908d94f4d473e763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c8ad674787669b9ae6184a05c893d7e
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ce588118ae6765f67e411222a2e0820c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_979215e17f78b02901f27edd6077d7cc
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2150cf91a7108dc36a2551d90730d74d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1174, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7b5965fea91eac1560ae8c2050b498a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1174, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_625946dbdf9db234f11a4829dda3f443(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e4faf2f6e23ce0683ca1fbcdb17c227
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 6, 1174, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cf530934e1cbb13e2ff465cc92f77ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40165fb2fa79b5603a3e78db4b8689d7
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 6, 1174, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_306a62051cb46b72f23072fc648533bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5a1416a809bf9b5894d068cb38e65eb
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 6, 1174, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cf3fce9d89439b6750243299e405a456(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_01f837ac83a5e47d515fea9deef58155(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_109a8ba4754251f080a18f7b1b7ffc09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 256, 8, 16], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7a77ef43f743c7b24123e9dfd8052a5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 256, 8, 16], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7d10690c8a804086ebb2e5ce6a13408c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 256, 8, 16], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2d145eee10646f670ac75938f97aeb6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_70a73370ab479cadb37418edf2fcaa92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_66dd7a184e831d9935c112515d8cd90e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 116, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8518b9f08ae0f52915043e56a5da65c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 116, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b33c6cd86844c6cc8fe9f94d8b640b1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 116, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d9ae1bc8d30d35d45fafe75fc599f0a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 116, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c3e9b15f806de99438dff0c79132ab4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1174, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_15a6f5e6a41baa7bca2d854a8373df69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1174, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c9d52b6617c821108eec2c637d4bad24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a9a402c476057416e05e8445bee5c6d
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 1174, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_864ad8ea1a1816ef4e4c4232cc426065(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_844745798fe7c273a18041fb38edd8d5
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 1174, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6f21b2f8f62da22571dc6a99a23bc414(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e5b40b387b25b5cb4c2c72230901880
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 1174, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_637636251167c0b4ed5f04b94cd71fe0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89a973be4b6d2c0af1abd9223ffbc7aa
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5177d27538399ac8b5cfdd5dbb45c531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c64e093bff901e5ad956ab61e94ebb9
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3e9de423f3e45706c2b21b1ad1279937(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 768, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d215fa722fa41f5bacb24166ce5c5963(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 768, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_25ce0a5b7a795e5dd72db9d10e591958(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 768, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c561ba07167a17fef0b2726851407257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 768, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_20e5db61e7bf92499b9c5dc056a1f621(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 384, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_28795fbe713c9afe46ab708aa0fc1b51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 384, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a7605efa00ebf702b4764b71289454a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 384, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b02fd349fba6d4dc097cf49a995dda9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 384, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4791c996e5f42eb2771b9815a8fd7e5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d58142a1a1f5def4096a90b1144972b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7705834c68ee81af9711c7c14d255e09
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b7e43454586a4cad186cbb2dc9ca27f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70d5d7c34b6bb343ee79b466dfded468
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_137c8f7db114adb08cbcbf250f9f60e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672d7ca116cb411bd493f55c9c11b420
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c7cbefce2c84a8c17dd924eddac2e580(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beac1656c826b0066c754a082454b7b3
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0c0d65ac4d0e3a2141a2a690f2d7ca03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c099470319cfe3f7120a1a3acccfdeaf
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2504c5aa95b2942d2fd7bec9b55a934b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7bae9133bc3890f818f8f67d01cc84c
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 56, 56, 96], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_16006aa6625ed2fa81d4af1cb4619b50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ddd68c4673164cda3ea7dd09f0008629
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 8, 50, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ffe43aee95632b14effacede020488d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6cb5625ac551fc45d8f35c17e21c2bc0
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 8, 50, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dffeb0a6d91a0eff8eba89de3981dc40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b306c0f3e6e955ed468d2cae2e394bb0
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 8, 50, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()