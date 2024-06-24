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
        return False
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



class PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3194e929567a2e665d1a4f19f2a13c95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 24, 36], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9f1013f0d927b7b214fa6beb233fc40c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 24, 36], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_655a8853d524fc358a7a115c0cdb6310(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_84f9967d87951d387443a92cd2695000(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d755a5918f1fbe9c57ad683e82350321(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_12cb482bf8691ca322b0439aa64fa26e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_8318d1ff1b398b6f29c70db7fd5d5774(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_99911b2e15b6a94e8bffc6ccc804b1a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8318d1ff1b398b6f29c70db7fd5d5774
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_dc5cccfcac4003e23fe5e6e7c450b89e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f2ec23148ff2f4e13eaf5021918c8ff3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc5cccfcac4003e23fe5e6e7c450b89e
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_af389d70137896b99943816cd6c674ce(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_324c9c6a0129d4cfd73171c180812979(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af389d70137896b99943816cd6c674ce
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ef03dab723665c0d72bf7c74ff9a09cf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b2c28bc9563d9f57cf926ffda54860fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef03dab723665c0d72bf7c74ff9a09cf
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_288f99b138d13f6e0f2aecd66a63c62c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_363c270d024cd677a49ab8ee9ae072f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_e6e81743cb1d0700dfd04ea418e38590(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dc2dc918341c7bfa8a5400d1ab8852ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6e81743cb1d0700dfd04ea418e38590
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_11a63fee878a52c1569d0df1fe557261(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([3, 54, 3, 198, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1a05ebec92871583e4b15bfe95d36ebc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([3, 54, 3, 198, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_14228a7783473c91299adab1dc635a14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6e81743cb1d0700dfd04ea418e38590
    def get_inputs(self):
        return [
            paddle.uniform([3, 54, 3, 198, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d42c3f00a30d9e2cd5c7ce92c7222d32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([2, 1960, 4, 16, 6], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1260f46fab32008f8b80b46b8499b164(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([2, 1960, 4, 16, 6], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_505a7ed3df6b7f741a8c8f38d5878e8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 192, 7, 7], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_284ff7a4783e41e1130ff3d300a4552c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 192, 7, 7], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_78c322cb82df24a501aab30f9f4b815a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_58fc50bd6605c2aac81ba244b9a19fc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 192, 7, 7], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [3]
        input_2 = [4]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3f79ac79b501ab67842bd7c65bd92832(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 192, 7, 7], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_98e691a2da41d304c1de705e4d56d986(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_43193033202c1a35556f209ac7b4e112(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6575f90c58cffdad43733b2950306ac5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 112, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d13e18b39d1d4e44429380f844436254(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 112, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_909e2021bcb467262b42493e339163dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 112, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_52dc736bb9cabc507941ae0c3b456c5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 112, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_82cbf80e51ce8c335681be9689f407c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 128, 16, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5db96438879f0b751cadd0abb760eaae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 128, 16, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8d459e354848e670451619ce4c913a4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 128, 16, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c989e7fc9e4eb61a31cdb2b6d0adba7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc5cccfcac4003e23fe5e6e7c450b89e
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c54f3dfcd72ce6c6916b464a2a209b4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af389d70137896b99943816cd6c674ce
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a2454970f6fe6c27f8e1c3963273d986(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef03dab723665c0d72bf7c74ff9a09cf
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6fc51e78302b45dec9987523bd9a1c5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3549, 76], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_abe755e5fffb4c095ce8f808acbfc37f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3549, 76], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8b3eea8f09f4344c03f72566285073c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3549, 76], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_563fde9da9e706dacbe077b3a9406492(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 576, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c01fdc0a273e2fbad7b0a26c69fabafc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 576, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_633f36fec58c24822aacfed78b125926(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 576, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_15e6016bd8ea9d78374d3e20f2958a6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 576, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a49dfeb166019a44cf9347f7bbf26e0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 288, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1c7f277774047da0110033a1a65e1cb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 288, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b97013ef29f4d93db1f02da58268afa7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 288, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cc5e23b50c075494b93231b8bb9359f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 288, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bd43469e7597980ecfbdfa50884e97e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 144, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4bcb0a6211ad46afe2c22ffa667a8c0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 144, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_caa5acc5ca31b766914298a0fe18fce0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 144, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4fb14f5c079e46d10f152c01ea5abbb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 144, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d6ab956fb65da46b6d707074ec6a998c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0c9ba19eaec8b333c387b9e9904546b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6ab956fb65da46b6d707074ec6a998c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.05675641819834709, 0.3304600715637207, 0.06396379321813583, 0.41534051299095154]], [[0.42963168025016785, 0.3483187258243561, 0.27696508169174194, 0.3411482870578766]], [[0.36815306544303894, 0.2565034329891205, 0.21405671536922455, 0.18817031383514404]], [[0.04792821779847145, 0.006738851312547922, 0.360918790102005, 0.3956049680709839]], [[0.4340068995952606, 0.4289870262145996, 0.4037555456161499, 0.32522040605545044]], [[0.1019599512219429, 0.42194366455078125, 0.4615171551704407, 0.21647119522094727]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_7d197a762c9df9dab5c6af71a4758a4f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4845725ac96179cbc9eae6d0d5582f1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d197a762c9df9dab5c6af71a4758a4f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.05675641819834709, 0.3304600715637207, 0.06396379321813583, 0.41534051299095154]], [[0.42963168025016785, 0.3483187258243561, 0.27696508169174194, 0.3411482870578766]], [[0.36815306544303894, 0.2565034329891205, 0.21405671536922455, 0.18817031383514404]], [[0.04792821779847145, 0.006738851312547922, 0.360918790102005, 0.3956049680709839]], [[0.4340068995952606, 0.4289870262145996, 0.4037555456161499, 0.32522040605545044]], [[0.1019599512219429, 0.42194366455078125, 0.4615171551704407, 0.21647119522094727]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_08fc1171e75c83d4c78e98e6b04b1aa5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8010d9ea85c50ee3453795d095846564(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08fc1171e75c83d4c78e98e6b04b1aa5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.05675641819834709, 0.3304600715637207, 0.06396379321813583, 0.41534051299095154]], [[0.42963168025016785, 0.3483187258243561, 0.27696508169174194, 0.3411482870578766]], [[0.36815306544303894, 0.2565034329891205, 0.21405671536922455, 0.18817031383514404]], [[0.04792821779847145, 0.006738851312547922, 0.360918790102005, 0.3956049680709839]], [[0.4340068995952606, 0.4289870262145996, 0.4037555456161499, 0.32522040605545044]], [[0.1019599512219429, 0.42194366455078125, 0.4615171551704407, 0.21647119522094727]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_bd39f23a5efe05254e03d60f3fc3356e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [3]
        input_2 = [4]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0daee373e870885c6141bc61c1cf4ff8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd39f23a5efe05254e03d60f3fc3356e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.05675641819834709, 0.3304600715637207, 0.06396379321813583, 0.41534051299095154]], [[0.42963168025016785, 0.3483187258243561, 0.27696508169174194, 0.3411482870578766]], [[0.36815306544303894, 0.2565034329891205, 0.21405671536922455, 0.18817031383514404]], [[0.04792821779847145, 0.006738851312547922, 0.360918790102005, 0.3956049680709839]], [[0.4340068995952606, 0.4289870262145996, 0.4037555456161499, 0.32522040605545044]], [[0.1019599512219429, 0.42194366455078125, 0.4615171551704407, 0.21647119522094727]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c60482f635ddad4d292b655a3bf4f2a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8318d1ff1b398b6f29c70db7fd5d5774
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ff0ebf24c2811968b938165eebcf168a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 40, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d56370675a440070ed1d8de3e7fbbd11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 40, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3f2a2f2c8b87581e517b580cba32a686(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 40, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_aba20f6f4e08aa92d142842cd7b690b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 40, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e9e918737eb94fcbf51a6e5c094d5e11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 80, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3f0c2a174bf7eefb3fa97318ba337cba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 80, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1aa59bd3db723b733d96e05eb96523a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 80, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fe1f991aa0f29f4eb9a013b6ec6e672c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 80, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b627bb2e750eca5f702b017fcc5d15c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1024, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7f86a7ab7becea6c465c908fdd5ec7b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1024, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_994e14426e1e005b78b0391b28e826ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5fa73e1e3e9d746f9fe7bc7fe5145c01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c3f03409353aea1d34127525f1d6959d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([128, 320, 8, 4], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ddafc5ee08de0159bc88c9b91b5c7b84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([128, 320, 8, 4], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e22a0b3b84fb35acdb3eb208a7356618(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([128, 320, 8, 4], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_54a2d5ec4e6517146bcce0e77b3d3b10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 49, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fcb6ec80da738734d47c9cb8864d9283(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 49, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_19279ba78765fd5d58fb5d30a2a0bdbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 7, 7, 768], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_21f2872b21c77d83e148b26049353205(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1c577641aa6afb48b8f020cb44610778(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_927eb710e97679742110fffb55513838(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_05fb74383408eb2491b5ccd00c27537d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f55dd9f863edfcab9c7be4f604379da9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a046a74352ae787ee7e063134086c465(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7cb60529fa89a23124773b91a8d8e528(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_229e46a5233415798ff6e4e2c99af11d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_89c1446f831d9b9085029c8f26860a0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 128], dtype='int32').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c3b08591e973c19f140736e207dd395a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 128], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_84f203abddb16517ea794bd4f956d33e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6ccd6f1f65acecf92ece9922dddf04e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ef5a907bf1cc6ddf228329d49969454d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 80, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ac518d6005f28eb2c3a9e7cbd072055f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 80, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_da5a27f2e7b5b3f524c485d65dffe360(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 80, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_717e1cc0f8c1c7fe6e47faace099fe12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 80, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0ac622e9b855e60a6aeb39ba13432d9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_51c334a637abc0dca21cd1287e981eb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_87db6b1261faac135a9ea9007a94aa78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6e81743cb1d0700dfd04ea418e38590
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_56d0e4f73b9bf99efb9b2b9b4bcdb608(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 7581, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3d962ea66e4668a66c939bf048e7f17f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 7581, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f47c555d85f7e4c27ddad15faa114e38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 7581, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_019f9c66cbc1b3de5343b06bef5cef7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([528, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_697e537063b6eab991c7d583d92f5b02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([528, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_519e9d6bfe8ac8476d457a12bea307ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([528, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ea41b9b1e984c5a84f5eedafa2853f15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([528, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a74fc244bc94bdf70bbf0fb47a86c774(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 25, 38], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_aba481eea4b5e6a64b60440ea78974d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 25, 38], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7934d516b845c8dc3c225add09f157e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc5cccfcac4003e23fe5e6e7c450b89e
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1ceb7ab507a921cc412da158ce8c5224(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af389d70137896b99943816cd6c674ce
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_27f471226d2f446745e935b45e163051(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef03dab723665c0d72bf7c74ff9a09cf
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_54414d8747a962e2e2125d66a1b65248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([12, 288, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7961c9bbf1f0ba88dea2fddddb5977ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([12, 288, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9234df5c1ef4086ef3a86bb8ab129658(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([12, 288, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_67f79b293763dbd5106e6cd964eb4b28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 320, 8, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6229903bcca8b63a71529e149c165479(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 320, 8, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2adb4c4d3dd3cd607ef0286a95478162(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 320, 8, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_20087e03b48fc858dbd16599e7acdc2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 4725, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e276ad72368896790ed1849b19f68f94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 4725, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4d1d6af991b89bc29e2977cf659619e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 4725, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_82425117932284a4da1d1880c891a144(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 160, 8, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e0b6bf7ce5a9d8ca6a7e37c78e7535f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 160, 8, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_547258964c98cea12f32b15648a11941(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 160, 8, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5be86beebb2b0f1f124a2490092a0655(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 577, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_efa6abc161854a78c3eba7bb69e28b1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 577, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3266d5899ee1ac8c2e2d6031131c561e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 577, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1dc4df4f4903039198f50221c18c53e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 577, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_25be7d120ba336625dab5a1649279fb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6e81743cb1d0700dfd04ea418e38590
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 577, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5d052416665deb50b19f8ab15be0e147(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_57b66bb5d1da58518b74ed042ab8d43c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bb9c14bc05258f185ac0ca230cb3db62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 14, 14, 384], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0fe8a6f35c76d6ac9250dbc8f548474e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([64, 64, 32, 8], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c39572c9e9dd4da1c9cfec825aa82401(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([64, 64, 32, 8], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5fe94bfc8c76b31a0222a7fe21080929(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([64, 64, 32, 8], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7f78f46eaadd1b31c5bf16e23d6a54e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([2, 10, 6, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e05b17c18638057f4bce7a85b9454543(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([2, 10, 6, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4a94e7ce45a386a03613681641e1cbc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([384, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b7c6f519a22657b67d052d348420aeaf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([384, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c714c2391af0248596b815e1d4ff70b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([384, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4d877bf57c40aaa3a83f7218b4ebc82b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([384, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1ed3cba9d7ec326c223ad09252ab2dee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 784, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_eca6ca7eb8f980aa860dcc298e045f71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 784, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6656f17b2316a05d244292949731c7c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 28, 28, 192], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c8749562cf480ad400e4fb1085109104(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc5cccfcac4003e23fe5e6e7c450b89e
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a66e4728e0077849392e27a124ea456b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af389d70137896b99943816cd6c674ce
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_debd3bdf050935c2ece76c2df1deeb55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef03dab723665c0d72bf7c74ff9a09cf
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7b6652e3994576270da6fdad89d58173(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_95b9d23c77421bcf539eb2c370e745ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_84f203abddb16517ea794bd4f956d33e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6ccd6f1f65acecf92ece9922dddf04e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8e0ea7b8b62a6d63972adec48988db2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 128, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8ceee9011f9341d501f3ee6e5272af9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 128, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8d1f719e0502c69a8291f1c69db1bdbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 128, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5c22b46430911540a8cce79bb967103d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0b99fca1e0790bb6532d75af31c50044(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0ce4c15de31d5bdba81c58e03ae0773f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 8400, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6f1adb329c33884eeaae277468858327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 8400, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9f7d3dd749f240324035abcb20097586(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 8400, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_99dd0a603fbdb2892404113a38ec0d1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 112, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_12c042ce47aaa5cb9ce237b4413671ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 112, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a6ec9edf48a36be256268be0739fd69a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 112, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fe16cc3165f79f8a81a5d93d6214bafb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 112, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7341f30c75a05643eda5255b10e400bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 20, 30], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5ca3b13bb902de312994d2330a55e803(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 20, 30], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4d2ed1f6afe296a7c4aa8886e9468cf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 40, 28, 28], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d11623144d06f91db23b984b748ccd1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 40, 28, 28], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_efa55fc7f6f88a6483457f60597ea634(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 40, 28, 28], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd1875aca183b01e8f1f5ef8b87fbced(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 40, 28, 28], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_065240a26ef85eb4ce8896b994361d28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_03ef8d86bfce57aac2f9c110ba333614(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d387262ed8a5e267f90da4b30e0b9487(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 56, 56, 96], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a626099936c4ed567a5206906187520b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 24, 56, 56], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_054f0b294057e1383c104bfb0e445ad5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 24, 56, 56], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_101049710ac4373fd218f98840f014b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 24, 56, 56], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2d33fa21fbe2c340f82f4d8bc245e2ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 24, 56, 56], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2160ca1e3e71349ac1a502dcc21ea410(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 160, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d1addbb16e6a8e20314d52d889ce1073(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 160, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2906b9de08205cb70aeb3c74956fa487(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 160, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_224889cc0865c5b7935102fd18ca2ce7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1a3c1c6416674325a0769df98b7493b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a03bf2febdf43635db3da4a63348d271(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3327f47a42a976c51ab93f5321c74f36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f55dd9f863edfcab9c7be4f604379da9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a046a74352ae787ee7e063134086c465(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7cb60529fa89a23124773b91a8d8e528(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_229e46a5233415798ff6e4e2c99af11d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7c8cd4ca75a51aaaacea09da49eb2830(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3549, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_84a62ec8e51576364f0aac53323bd58c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3549, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a80df350716bbcd408e99a206502300d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3549, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e04773b902bf61052d6383fa93a40d1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 768, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_805ae1cf8f1e1ad18ef567476a2e4df7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 768, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_19989624090fd78511bfdbe171839aa8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 768, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_79db74fc1c465f46928ca89379482d24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1025, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f207ba317a8f2d1894161529c5fb0467(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1025, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_42576d3f20dfab99bbd5eef256f6e677(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 64, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_38ddf567c587490cd709512265426e3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 64, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7dfbf2635777d04bdc4e5a12d4de34a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 2, 640, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4aab08c8fbceabe3f6b637ded92cfaa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 2, 640, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_941dc93ef114fe16778de2e3acc5363b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6e81743cb1d0700dfd04ea418e38590
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 2, 640, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7ff6a1d9f583012f7532dea20dafe5ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([3, 86, 3, 198, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_85bdf8a0dc7dac3a2ec5b0d0ee4f2e0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([3, 86, 3, 198, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_825b9441318909376f5739ef9331410f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6e81743cb1d0700dfd04ea418e38590
    def get_inputs(self):
        return [
            paddle.uniform([3, 86, 3, 198, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b7381e912b8da77a017fc4f43107d26e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0f5cb0df636a5c41f366a790911a541e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_49256fce9294aaab063524a9bab72231(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_63fff7ffc41946087f676bd8d545126a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0f085ef50d207ca88f42ca2f7d0f8310(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 192, 7, 7], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b4cf1582cbfa8cb2da67e86e3e5d413d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 192, 7, 7], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dbf699dc4242dba143ca798a15df0fef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 192, 7, 7], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6d61653b21eff966796efe1092e92130(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 192, 7, 7], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_370838a22cf12842269ebf434a7a1cbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([20, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04319af5125e4f014a252bf43dfecb7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([20, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_61d4f67cdb45f6dd9fe9a8f5c390893e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([20, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_208789c450fb9fb86af0297bde75510d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([20, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b514ced15c1b5d1ea88d3d1575c42d58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc5cccfcac4003e23fe5e6e7c450b89e
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b8560e250c86051567b2b2ecf84893c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af389d70137896b99943816cd6c674ce
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_edfac21fde011367160c1e676e27e180(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef03dab723665c0d72bf7c74ff9a09cf
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e72692c421d9be092ec0b6489ac370c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 97, 97], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7892f35404cfbc250868422af0ede919(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 97, 97], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6f49f915d4437eea979a652eded6e994(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1, 2]
        input_2 = [98, 99]
        return paddle._C_ops.slice(input_0, [2, 3], input_1, input_2, [-1, -1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_39af72c18f8515f9f7f6b0741922dae0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f49f915d4437eea979a652eded6e994
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 104, 104], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 2], dtype='int64').reshape([2]),
            paddle.to_tensor([98, 99], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5fdc3d78f8beb3dcf6add271c69fddbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([2, 4312, 4, 16, 6], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0448ca47ddf1cacce66f1a29e2e532da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([2, 4312, 4, 16, 6], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c260f358ce6f3ea06d3dc35aa3aeb5ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_be8f716de5992b4e8d15313f8d761bc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c989e7fc9e4eb61a31cdb2b6d0adba7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc5cccfcac4003e23fe5e6e7c450b89e
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c54f3dfcd72ce6c6916b464a2a209b4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af389d70137896b99943816cd6c674ce
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a2454970f6fe6c27f8e1c3963273d986(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef03dab723665c0d72bf7c74ff9a09cf
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_30121899b51bacd2490dc9e15e71c8bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 49, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_223ee0703cfed028f7dbe66d897946ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 49, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b127430db3d4e88f5aef8f05792649f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 7, 7, 768], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9ae2d6648181717150ee37a8b4025725(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 1024, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b56d4854bf6ea86e4ca0c11afb83e2fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 1024, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_215190640398832a3a23d08c226d4b48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 1024, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b8bcd9fd5f4a2bdf1e88262800a048d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 256, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_776cb4cb34c1b634e32b15931d07c4db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 256, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ad8d125777636748f9c3654e2cb914c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 256, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c2b5de4e9235e33e15685f10b770d4f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([576, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5e638b3396ecb6fd8b2d2e7c6b6378a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([576, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cc894d62a72ce8c6824379d0c79e99cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([576, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e5679af55c7d1fa21a1b5b5020620b36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([576, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_737a45bce68d11a32958d9db4959f428(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([22, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bb351264ca34f8a8c14699f1f358202c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([22, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e323fa4317f764d34f6937651bf4a7cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([22, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f693e71cf63473877d1e70da422efe34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([22, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5e727cbd7bfeb3a657afc0d51e0a7b91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([96, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8c3b1c6f324caf4c9c209514a37a890a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([96, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_816d4e6d950a9e39206a8a3fb8f8baea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([96, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cee64d53afd2036451685e9a366b80cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([96, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_50545704bd4a86b05b0830c3d03bfe01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([12, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c873394a2e9dd150bebdc87b40a5e8d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([12, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_608cec7d9e0cc991776d83e680253668(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([12, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e5db201dd3290ebe5ac00c7b2f3ce07d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([12, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86a964ea00a5961272761378310c3bfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 256], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ba6b8faa931793e42ae38303620b8ca8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 256], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f43f9c18f8643d5de0758f9585027e29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 512, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f6abb332dccc8092695f9369c1dcb3e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 512, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_42576d3f20dfab99bbd5eef256f6e677(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 64, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_38ddf567c587490cd709512265426e3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 64, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6d315ce34b177ce0ada24af80610f36f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([6, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cc88262cccc1001f7475375a0df8856d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([6, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_470bfde13a5ac2858c5017983812e0b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([6, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_115735d8aaa9a0ff67c400c7a7e8ad46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([6, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b360c1d6c26750e19dea7642ad74c199(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_812cb804a39a3a25910e4419767aa522(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_930d445d3a3862895ee95b430b2d993a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([960, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8e8cf96227a28087af5cdbd6572cc93e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([960, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_31f53be899ea1492dc8f5b128899aacd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([960, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d043a6b3b099dc222f06857ac85de287(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([960, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_288f99b138d13f6e0f2aecd66a63c62c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_363c270d024cd677a49ab8ee9ae072f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dc2dc918341c7bfa8a5400d1ab8852ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6e81743cb1d0700dfd04ea418e38590
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f095c99658b42c5ac7a9d6632dd59e07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_45dfce148ecf24e2da1e496e6b1d65f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_28846be43ae960f6e7236554a67d1ecb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 256, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b886d92d501e88d543d07bdfd384edba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 256, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_28c289c69cf2b5a3b258819abc18be9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([2112, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9b32dee7b5fb95f2d5537a704ed67582(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([2112, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fc51db87b0cc81c6ec63670db55ab4a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([2112, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7b00ee86154de732816d836c19685f16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([2112, 2, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_97dde80e92404d22e781e30a7166cc3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 72, 28, 50], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_253b2805f946e58cc279c110872a59f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 72, 28, 50], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2b3ae6d0e38872a679a297bf61e229e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 72, 28, 50], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b628d1e8ed37f6207cd241a0e96929d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 72, 28, 50], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_25d912618ec71e4da9f970f81cff08ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6ab956fb65da46b6d707074ec6a998c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.4153806269168854, 0.29699423909187317, 0.3866505026817322, 0.3970966041088104]], [[0.19716790318489075, 0.27303647994995117, 0.3432890474796295, 0.36351412534713745]], [[0.23610685765743256, 0.294491708278656, 0.30934077501296997, 0.3172841966152191]], [[0.45234987139701843, 0.17155957221984863, 0.2550120949745178, 0.284310907125473]], [[0.10462033003568649, 0.2873287498950958, 0.3279913365840912, 0.1624995917081833]], [[0.38143959641456604, 0.4887605607509613, 0.005508876405656338, 0.40235456824302673]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e33cab7231fbe4f89165f9996afbe0ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d197a762c9df9dab5c6af71a4758a4f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.4153806269168854, 0.29699423909187317, 0.3866505026817322, 0.3970966041088104]], [[0.19716790318489075, 0.27303647994995117, 0.3432890474796295, 0.36351412534713745]], [[0.23610685765743256, 0.294491708278656, 0.30934077501296997, 0.3172841966152191]], [[0.45234987139701843, 0.17155957221984863, 0.2550120949745178, 0.284310907125473]], [[0.10462033003568649, 0.2873287498950958, 0.3279913365840912, 0.1624995917081833]], [[0.38143959641456604, 0.4887605607509613, 0.005508876405656338, 0.40235456824302673]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d94ed08fc6376bbf0f8ffa2dbb337b5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08fc1171e75c83d4c78e98e6b04b1aa5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.4153806269168854, 0.29699423909187317, 0.3866505026817322, 0.3970966041088104]], [[0.19716790318489075, 0.27303647994995117, 0.3432890474796295, 0.36351412534713745]], [[0.23610685765743256, 0.294491708278656, 0.30934077501296997, 0.3172841966152191]], [[0.45234987139701843, 0.17155957221984863, 0.2550120949745178, 0.284310907125473]], [[0.10462033003568649, 0.2873287498950958, 0.3279913365840912, 0.1624995917081833]], [[0.38143959641456604, 0.4887605607509613, 0.005508876405656338, 0.40235456824302673]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3f0594ed6603158e1a1c8f1bd804a3da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd39f23a5efe05254e03d60f3fc3356e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.4153806269168854, 0.29699423909187317, 0.3866505026817322, 0.3970966041088104]], [[0.19716790318489075, 0.27303647994995117, 0.3432890474796295, 0.36351412534713745]], [[0.23610685765743256, 0.294491708278656, 0.30934077501296997, 0.3172841966152191]], [[0.45234987139701843, 0.17155957221984863, 0.2550120949745178, 0.284310907125473]], [[0.10462033003568649, 0.2873287498950958, 0.3279913365840912, 0.1624995917081833]], [[0.38143959641456604, 0.4887605607509613, 0.005508876405656338, 0.40235456824302673]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_db13a8ecea7e566b51fc32ba53899c32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 4116, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7754004c8d64ae16ec2c661044629a6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 4116, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2e94e4af38824b0c563b746b5532b335(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 4116, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a9f271b691ef42a670cce3ae7d2e37c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a051d3073974810fbb5d48c6cc0c7f6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_7ee20b24d0dc5c0f8dfc8a85b251f263(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_30384cd161f78e57e9f820ceb70d75e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ee20b24d0dc5c0f8dfc8a85b251f263
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ab25d3d5c606d444873ae972ad49a88d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f641b362602828243e938fa24da95e48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab25d3d5c606d444873ae972ad49a88d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_206bb51737a40e49bcfd02c33c6f47a8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cfd51d1f58bb2de524face156d9b624c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_206bb51737a40e49bcfd02c33c6f47a8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_60c3a53dc70160884ae36824e2bb7ab9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [3]
        input_2 = [4]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_97d169986c5275c5bdce4d987b63d78e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60c3a53dc70160884ae36824e2bb7ab9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_5ea652a9b28ef535190e472ee073bfd6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [4]
        input_2 = [5]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c58678f18163b850b3f8608ca29ad89e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ea652a9b28ef535190e472ee073bfd6
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_fc217cebb3119779c263c0b26c1c0e92(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [5]
        input_2 = [6]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1110ab80b33859d164d441e490480e90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc217cebb3119779c263c0b26c1c0e92
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_65b631facec9038f7274627dd8b3d34b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [6]
        input_2 = [7]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_88dc24592c6840b78ea065b751f0b248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65b631facec9038f7274627dd8b3d34b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c1abc5a84c7b67e7998017479b702593(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [7]
        input_2 = [8]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5792c46a72a2eebcda19299bfbbd68db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1abc5a84c7b67e7998017479b702593
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_80551ea934c2d61c03fa6457d9ce1985(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [8]
        input_2 = [9]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5a5b1e3c6060456b800f01c61452e09f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80551ea934c2d61c03fa6457d9ce1985
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_a7dfb4ada84d78f903e5538507d3b7ab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [9]
        input_2 = [10]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7e4230d34fe2238b24baf8f55d4316a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7dfb4ada84d78f903e5538507d3b7ab
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_91e429e4f0dfc5c2769ccfdc4607b6fa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [10]
        input_2 = [11]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9873892219239feaec279b3e3b8221e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91e429e4f0dfc5c2769ccfdc4607b6fa
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f1262774c9edc7ff8f4ba3df5056d811(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [11]
        input_2 = [12]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c255339dc3a5e95bca5d8966d277b3fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1262774c9edc7ff8f4ba3df5056d811
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_b98c0cd022d0194b4c0832b0db60eac8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [12]
        input_2 = [13]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0fc78f2182fd602b39b6f4993e0fde82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b98c0cd022d0194b4c0832b0db60eac8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_aaac1a3979662b89e725d317a548ca91(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [13]
        input_2 = [14]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_59fec9bbfc0a94b60fb4eaeb522eea3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aaac1a3979662b89e725d317a548ca91
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6d52063c93809e6ba0fe1e83fac1cc5b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [14]
        input_2 = [15]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b9c39a21140a9935761e2af594b89c14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d52063c93809e6ba0fe1e83fac1cc5b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_33218763cd1a25505d53cd5b29035e0a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [15]
        input_2 = [16]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dfc4afa7670aa12b435c5d0d2dc21a38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_33218763cd1a25505d53cd5b29035e0a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_786c42a7e462a07279f1e57ddde082e0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [16]
        input_2 = [17]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dfeb6188aad3ec35a01323f37a19e143(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_786c42a7e462a07279f1e57ddde082e0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
            paddle.to_tensor([17], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6f768750d0804989fa348867c53b0bb1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [17]
        input_2 = [18]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f09b19d1b84888c151e98a369edc0850(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f768750d0804989fa348867c53b0bb1
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([17], dtype='int64').reshape([1]),
            paddle.to_tensor([18], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_aab8b254011b8ce61407c27f0f4c642f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [18]
        input_2 = [19]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1d896b7a1686bc06610e7736740c9f0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aab8b254011b8ce61407c27f0f4c642f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([18], dtype='int64').reshape([1]),
            paddle.to_tensor([19], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_adacbc4e820398794baec64fd98a3eb5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [19]
        input_2 = [20]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_83acbc44a935d2842116fa08c79b9ead(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adacbc4e820398794baec64fd98a3eb5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([19], dtype='int64').reshape([1]),
            paddle.to_tensor([20], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c129e38203897bdcdc39d76ab610abe6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [20]
        input_2 = [21]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1f95a333eb3eb4da511126e59b88b0b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c129e38203897bdcdc39d76ab610abe6
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([20], dtype='int64').reshape([1]),
            paddle.to_tensor([21], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_60c299b11353f17bec6c5760b2ec71e6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [21]
        input_2 = [22]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_466e59076048892c279a2d760b39baae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60c299b11353f17bec6c5760b2ec71e6
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([21], dtype='int64').reshape([1]),
            paddle.to_tensor([22], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_fd1b9dfb33b0bf8fc2956226a617717c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [22]
        input_2 = [23]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c8109e60bfcd9a6dc09ed553c4d25a62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd1b9dfb33b0bf8fc2956226a617717c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([22], dtype='int64').reshape([1]),
            paddle.to_tensor([23], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f598268b6c5d69f0bfc5945c4ef367f5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [23]
        input_2 = [24]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d0b8722c399b6491ce7e45f4eaf3f0ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f598268b6c5d69f0bfc5945c4ef367f5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([23], dtype='int64').reshape([1]),
            paddle.to_tensor([24], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6fc7f50d72d3a27bf2d77c6cfb11617a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [24]
        input_2 = [25]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4b8784b4b872a46b1becfb772d9db032(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fc7f50d72d3a27bf2d77c6cfb11617a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([24], dtype='int64').reshape([1]),
            paddle.to_tensor([25], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_97dfa6384f0e00834b4731f732250d7c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [25]
        input_2 = [26]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_44da6ae26a3352cafe8685c4f2a9919f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97dfa6384f0e00834b4731f732250d7c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([25], dtype='int64').reshape([1]),
            paddle.to_tensor([26], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_86b557c5989b47fdac5f937cb60ef377(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [26]
        input_2 = [27]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4cf2f64565cf447a519eef08c8117a92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86b557c5989b47fdac5f937cb60ef377
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([26], dtype='int64').reshape([1]),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_3bf328c3e1784a95d1c8c5174bbfc5ba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [27]
        input_2 = [28]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a52b3a27d113948de4dd9569343692a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf328c3e1784a95d1c8c5174bbfc5ba
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
            paddle.to_tensor([28], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_e5b66c62fc3e483a566edf5e8f981ea3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [28]
        input_2 = [29]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0fb3e1bf8004d84a0eae02458a199213(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b66c62fc3e483a566edf5e8f981ea3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([28], dtype='int64').reshape([1]),
            paddle.to_tensor([29], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_e4ba03ff4391abf9fd557ef65dd6325e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [29]
        input_2 = [30]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_56acd6d8c42cec70c17aad20b1ea5af1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4ba03ff4391abf9fd557ef65dd6325e
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([29], dtype='int64').reshape([1]),
            paddle.to_tensor([30], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4b9f54fa33b7dff41838d02ed1f33cc2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [30]
        input_2 = [31]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b95fa0307eb55da3a1494b0715fb40b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9f54fa33b7dff41838d02ed1f33cc2
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([30], dtype='int64').reshape([1]),
            paddle.to_tensor([31], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_9d35009155b9636cfbd28ba3893f672b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [31]
        input_2 = [32]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1224b45582934dfa6ba125eac362b552(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d35009155b9636cfbd28ba3893f672b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([31], dtype='int64').reshape([1]),
            paddle.to_tensor([32], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_e5a686f212739ed486f6ca0e7a1c731d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [32]
        input_2 = [33]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_188d0aadb598cbb9b5077a016ee6eea4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5a686f212739ed486f6ca0e7a1c731d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([32], dtype='int64').reshape([1]),
            paddle.to_tensor([33], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_7edebd0037dd4b4d9b8bb90546382d1a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [33]
        input_2 = [34]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e40d9ef18f7db5ae694a0cdcb961f5f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7edebd0037dd4b4d9b8bb90546382d1a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([33], dtype='int64').reshape([1]),
            paddle.to_tensor([34], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c769d7c7b73b95aa762264fb20471cae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [34]
        input_2 = [35]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f4cd71a4a506f90d970d16410af8d6c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c769d7c7b73b95aa762264fb20471cae
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([34], dtype='int64').reshape([1]),
            paddle.to_tensor([35], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d1db492ab74727de620b8a48a35c176b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [35]
        input_2 = [36]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8fa1c77412b38aed1213944b1bb6d9e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1db492ab74727de620b8a48a35c176b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([35], dtype='int64').reshape([1]),
            paddle.to_tensor([36], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_43d5b23888ac8f77827eaf5ea24e488c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [36]
        input_2 = [37]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7c6a27ca66044adc7e53249214fac812(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43d5b23888ac8f77827eaf5ea24e488c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([36], dtype='int64').reshape([1]),
            paddle.to_tensor([37], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_cf0e593fb3d8c20762680d907541d9ca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [37]
        input_2 = [38]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_611796ce06c7633cf692b2326e1f90cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf0e593fb3d8c20762680d907541d9ca
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([37], dtype='int64').reshape([1]),
            paddle.to_tensor([38], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_b8f3e6a0312f63512f7e28c470bd5b42(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [38]
        input_2 = [39]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f35a1cebab712a6f0546921f27aa9d8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8f3e6a0312f63512f7e28c470bd5b42
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([38], dtype='int64').reshape([1]),
            paddle.to_tensor([39], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6cb44aeab8767351869c59ba5c722f9c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [39]
        input_2 = [40]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_652ef71d25df8424d4c5395f3e22305a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6cb44aeab8767351869c59ba5c722f9c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([39], dtype='int64').reshape([1]),
            paddle.to_tensor([40], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_55e97202eb36c293cb8f7f714219eadc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [40]
        input_2 = [41]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a51e7c8ed09cccb0aa11ea1336c54330(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55e97202eb36c293cb8f7f714219eadc
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([40], dtype='int64').reshape([1]),
            paddle.to_tensor([41], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_2fe4027be3f35625d7febc50a58d4429(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [41]
        input_2 = [42]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_907e8743e23b33ffe7033d9539c4d3e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2fe4027be3f35625d7febc50a58d4429
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([41], dtype='int64').reshape([1]),
            paddle.to_tensor([42], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_aa07965657440e2272b2339a117b9b51(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [42]
        input_2 = [43]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3d4582c94aa67c6033052bc3958e8058(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa07965657440e2272b2339a117b9b51
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([42], dtype='int64').reshape([1]),
            paddle.to_tensor([43], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_1c25c4e732671e4d40e22bb7b9fd7c26(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [43]
        input_2 = [44]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_15654199ceb342523d8ca2232a6eefc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c25c4e732671e4d40e22bb7b9fd7c26
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([43], dtype='int64').reshape([1]),
            paddle.to_tensor([44], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4e94d2aba926c9fa13ac0e1831830c67(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [44]
        input_2 = [45]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8ee233821792cb08ad8c6a2f90550275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e94d2aba926c9fa13ac0e1831830c67
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([44], dtype='int64').reshape([1]),
            paddle.to_tensor([45], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_3a31b0d0b621cb336425463df47f5a24(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [45]
        input_2 = [46]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7c4fcdacc1cf0b4907a610cf1ae9f74b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a31b0d0b621cb336425463df47f5a24
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([45], dtype='int64').reshape([1]),
            paddle.to_tensor([46], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_2d37c28e6918cbf046fa5642b8cd12d8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [46]
        input_2 = [47]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2e240f2740039fa8222b0768db5ef25f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d37c28e6918cbf046fa5642b8cd12d8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([46], dtype='int64').reshape([1]),
            paddle.to_tensor([47], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c334926e6aca226b4a3efe04f7c96842(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [47]
        input_2 = [48]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2383244efb7b322b124a8b8950afa2c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c334926e6aca226b4a3efe04f7c96842
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([47], dtype='int64').reshape([1]),
            paddle.to_tensor([48], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_990862c82a422dc21fe38251ec184b68(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [48]
        input_2 = [49]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_38671d3651e9291ff9ba48fa5e736bc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_990862c82a422dc21fe38251ec184b68
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([48], dtype='int64').reshape([1]),
            paddle.to_tensor([49], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a9f271b691ef42a670cce3ae7d2e37c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a051d3073974810fbb5d48c6cc0c7f6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_37920c6c9dc094456212322c7e793ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ef6764fcd38eeecf59a75d4debf01309(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f55dd9f863edfcab9c7be4f604379da9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a046a74352ae787ee7e063134086c465(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7cb60529fa89a23124773b91a8d8e528(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_229e46a5233415798ff6e4e2c99af11d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_065240a26ef85eb4ce8896b994361d28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_03ef8d86bfce57aac2f9c110ba333614(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d387262ed8a5e267f90da4b30e0b9487(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 56, 56, 96], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_80df9300ae231fef6afaaec9275c30bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 960, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1afcc5c17161c076273130e73d5f20ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 960, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_31881604baadaf0338e180fb3090b42a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 960, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8a4f4d0ff88f7964ef67b96c9ab81335(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 960, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f90515959538e5558b3fff2f205bae43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 480, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e3b9b1a142401ac89bf9b301f27a0050(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 480, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fb656baa03bdb279e2d295f4fd5fbb50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 480, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e89a295aa67419717541f9b9f747cc14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 480, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7797fc39896425d9b2734176ad51ace2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 240, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b9dc3828de6da66aa260433ebc2fc7cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 240, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7fcd0d540799a77bf7386615434aa314(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 240, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cce4de90c2728c4eb24b9f53b819afb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 240, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9b250bb37f41e52a4f64f4265909a32d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 6069, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6115ea687633d5c6167fdd43916dea64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 6069, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0ba87a2ab29944853798ada6564bd364(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 6069, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a955381c7b3dd1a7cba9ee2da4f54e00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 784, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_78bee08152c619010c1d374d76c62eee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 784, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6832b635092cfa59e2614ebfe150426c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 28, 28, 192], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_deff5c9aa12110e193a8f95907c7dc40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 1, 1], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c8749562cf480ad400e4fb1085109104(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc5cccfcac4003e23fe5e6e7c450b89e
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a66e4728e0077849392e27a124ea456b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af389d70137896b99943816cd6c674ce
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_debd3bdf050935c2ece76c2df1deeb55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef03dab723665c0d72bf7c74ff9a09cf
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4e39643ad0e2a3ca2344f1df88ea81d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 8], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ccd20b82209aa3925dbe0831467a4021(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1025, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_324942a0c9d72196e35a2ebbecdd586b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1025, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7f1ebe5e55d3f91245ba4141acde1b97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 6, 1025, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f290d55eda5c74f40cac5c8f78f5d1f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 6, 1025, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4fe5fe905bcf1c8b06164659708fffe0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6e81743cb1d0700dfd04ea418e38590
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 6, 1025, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_03e8168d8363f28ed1dda2b9a722b2a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6cc277dcdabfbda3e8fa5c3e0d31e75e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9c3a0dae9c7784964d0f682058fbf242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_438533b69eca146526c0301ffecdd566(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 24, 56, 56], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_16f153d1f549e109f0860ee5ad37f8fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 24, 56, 56], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0572f39bdf4ee908b3bf414ac7a72e8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 24, 56, 56], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_26eea2363da50a66ab990e159318853c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 24, 56, 56], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8226eceb82e944be6e0fe77bb5aaa33b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_67a6949f5bf49e16d0226d9cfd17d1e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f6aab5fc05061ad2fabd464e3fe402b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 64], dtype='int32').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_af92b38ecd7d234913d5761fadea7bc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 64], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_57f5535a3ad0c98b795a6b75644a1128(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 1, 1], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_deff5c9aa12110e193a8f95907c7dc40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 1, 1], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5d052416665deb50b19f8ab15be0e147(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_57b66bb5d1da58518b74ed042ab8d43c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bb9c14bc05258f185ac0ca230cb3db62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 14, 14, 384], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ed3dfa4877211f57491caf450b47743f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512], dtype='int32').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a7cec52f1e68f19ae80067f9214b3c49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_67f3f81d46b439934eb5cd2a50251cee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 512, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1e5b65e948704d1c18a17b5387085845(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 512, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_129510356b49fb6a94f593fd11ba86a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_594ce056ca835126eaac4525a56fc631(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_129510356b49fb6a94f593fd11ba86a6
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 7], dtype='int64').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_b3f492c9a371b019e91c744b7bbdc2d1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ece635770cfa113eee5836744c56763d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3f492c9a371b019e91c744b7bbdc2d1
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 7], dtype='int64').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b879cf836292a2190fc3275990a59a2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1024, 256], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_67dba9fdc085f73912dfd3600e3582f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1024, 256], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8d30c0c25efff1e0a8d462a03802019d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 1024, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_24a3f06e3829350b86d455489f97bba8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 1024, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_acd4133ba829195b68e1d5d20a795e90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 64, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_14c5adef3d28cd55a3afd91e3862feda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 64, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_698a09a761091d40ba5a16f979317be0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 64, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_84f9967d87951d387443a92cd2695000(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_12cb482bf8691ca322b0439aa64fa26e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9d8e71aa001e49f25fd32e64e0fb39f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 464, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b8ef2a78e754af9a1448b39832b48b70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 464, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c748b986f6205695374954f36b918c4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 464, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1ecf53216e1ec7ac65b966830f756203(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 464, 16, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b514ced15c1b5d1ea88d3d1575c42d58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc5cccfcac4003e23fe5e6e7c450b89e
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b8560e250c86051567b2b2ecf84893c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af389d70137896b99943816cd6c674ce
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_edfac21fde011367160c1e676e27e180(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef03dab723665c0d72bf7c74ff9a09cf
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bc540b86cddf6e99dd387dcf48ce37b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([3, 54, 3, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_87f11edd2723d998d706dd8c9658fc40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([3, 54, 3, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ed0b63f81234a2e40b1f6d3f87414aca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6e81743cb1d0700dfd04ea418e38590
    def get_inputs(self):
        return [
            paddle.uniform([3, 54, 3, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_224889cc0865c5b7935102fd18ca2ce7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1a3c1c6416674325a0769df98b7493b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a03bf2febdf43635db3da4a63348d271(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3327f47a42a976c51ab93f5321c74f36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f55dd9f863edfcab9c7be4f604379da9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7cb60529fa89a23124773b91a8d8e528(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_229e46a5233415798ff6e4e2c99af11d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_224889cc0865c5b7935102fd18ca2ce7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a03bf2febdf43635db3da4a63348d271(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3327f47a42a976c51ab93f5321c74f36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b2f71462f3c7478c78e4581b5d96a1b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 1024, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_69a53632298f2c9c1b62eef562062679(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 1024, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_99911b2e15b6a94e8bffc6ccc804b1a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8318d1ff1b398b6f29c70db7fd5d5774
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7934d516b845c8dc3c225add09f157e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc5cccfcac4003e23fe5e6e7c450b89e
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1ceb7ab507a921cc412da158ce8c5224(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af389d70137896b99943816cd6c674ce
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_27f471226d2f446745e935b45e163051(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef03dab723665c0d72bf7c74ff9a09cf
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6d3cd5fd2b9ec2d1b235ff74348c2d82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 256, 8, 16], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_37dc7254713816c508d0c9daea359e4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 256, 8, 16], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f7c41019e1bc6fd65f3d467304c9ff4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 256, 8, 16], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e033907e4ff9e796fc484c908f198c86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_808fd7d99bce409028234f533aa1378a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_68292c1f37c3411379b185141269590a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 14, 14, 384], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f428c3727ee3386739684c0adf14b19f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_022dfb5bfbf75b8eaac290b9c2d9e655(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_da05c81f7ed576c676c8b314480acf3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1d332babaa3fd98d27da977d6f9bcc93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_975e370893dd0813f224e460f4343d3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 144, 14, 25], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_de42173e46909acb77bfc2f01605b9d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 144, 14, 25], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a30d78686da9e0405e1250d96166f2b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 144, 14, 25], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_55a80e119bfaee67975e57a8945aa7c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 144, 14, 25], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_49256fce9294aaab063524a9bab72231(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_63fff7ffc41946087f676bd8d545126a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d24ab48cfefc2a414a689b0608122145(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 128, 1, 1], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d9608434aecc389d518d2a016cf31f6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 128, 1, 1], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_53f2d292b8f704e76509406420c063ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2c4fe2e3b8be465edffea9f38bccf4ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a2ffa2b28e11ef87d2077af5a2de7252(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 56, 56, 96], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0ac622e9b855e60a6aeb39ba13432d9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_51c334a637abc0dca21cd1287e981eb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_87db6b1261faac135a9ea9007a94aa78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6e81743cb1d0700dfd04ea418e38590
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9f025e6f0100bb9f36d3068ed53a3135(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7e072bea3dbf2fc2a56c17bc069b84f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8226eceb82e944be6e0fe77bb5aaa33b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_67a6949f5bf49e16d0226d9cfd17d1e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f6aab5fc05061ad2fabd464e3fe402b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 64], dtype='int32').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_af92b38ecd7d234913d5761fadea7bc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 64], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cea1d1aed214b56f80b3169ed85053ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 2, 200, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_adf7722efb444526d38cfa4c0ae03288(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 2, 200, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_26648dd8e9482ba63ed6f7067b6edf43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6e81743cb1d0700dfd04ea418e38590
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 2, 200, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_088905a321bdde7f77758cd37eba0497(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 9261, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a51ed8fe3b0fa03c09f0c14ddca7fb56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 9261, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7c735fcfbdef6027b4ed0ccdbd7b80c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 9261, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e372ab65a5501db643988ad2a4243065(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ee20b24d0dc5c0f8dfc8a85b251f263
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fa29f91d4a852cb96797391dc32b146f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab25d3d5c606d444873ae972ad49a88d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0ee40801f0cd521a5b721fde827d5b75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_206bb51737a40e49bcfd02c33c6f47a8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_74c0692ccb0d3d87f7c43235de7567a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60c3a53dc70160884ae36824e2bb7ab9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_350bf857c14e39042a1a9b33d63abf77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ea652a9b28ef535190e472ee073bfd6
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cdfb7e3cea2bfeab5e95e08b1c85f407(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc217cebb3119779c263c0b26c1c0e92
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b5c2490d29196c0096ad6de0dfc07ad3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65b631facec9038f7274627dd8b3d34b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_aa8eb86d6c47f45504298abbe2af05a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1abc5a84c7b67e7998017479b702593
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5d56aace9bb703f84688555130354bfe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80551ea934c2d61c03fa6457d9ce1985
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_99b407e5064119b09a6b49fe42553d5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7dfb4ada84d78f903e5538507d3b7ab
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_483beb8a31506bc2ded6e08324ff009b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91e429e4f0dfc5c2769ccfdc4607b6fa
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_10e573d8668cbc798f5152cfa9278a0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1262774c9edc7ff8f4ba3df5056d811
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ca465e1eed0992055064a5d941760524(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b98c0cd022d0194b4c0832b0db60eac8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fa24273c452c9b66eb55c08a5cf46bb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aaac1a3979662b89e725d317a548ca91
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_db3da9d2dd5e420e29991e93da0cadb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d52063c93809e6ba0fe1e83fac1cc5b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_252dc8767aa61ac3734b07a3f6e52df9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_33218763cd1a25505d53cd5b29035e0a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6caeabfe73313717a620419730370a91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([2, 22, 6, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_99b7dfec0de020a2f1b5b85e607d7c42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([2, 22, 6, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b2af53c36e413cbcc380398d1475faee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([240, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a42870d439a58a95ddb1a382e895eba8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([240, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_939e0cfee566e62ce07ee60e8975b0d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([240, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_117232219e9572369bc75cf98a4e6ae5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([240, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7370c77ec47e606aec5a1160e940efa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e89ac373dbf26a985933b06c605ab4a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a447e32364a4226f421c3c04321a4ace(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a18ff75cba6e5a669971d0b956077b22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 32, 144, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b7381e912b8da77a017fc4f43107d26e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0f5cb0df636a5c41f366a790911a541e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b7381e912b8da77a017fc4f43107d26e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0f5cb0df636a5c41f366a790911a541e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b7381e912b8da77a017fc4f43107d26e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0f5cb0df636a5c41f366a790911a541e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d0d8409b8091668cd9a65a211755ecf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2048, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0e53fa2d386a081ba4fc95e0eefb2e42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2048, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a955381c7b3dd1a7cba9ee2da4f54e00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 784, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_78bee08152c619010c1d374d76c62eee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 784, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6832b635092cfa59e2614ebfe150426c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 28, 28, 192], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7f3da502c070c1f3292aacd6beb30939(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc5cccfcac4003e23fe5e6e7c450b89e
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_899c745138d6a89963cd276a4a2ed280(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af389d70137896b99943816cd6c674ce
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_837eda0b4f643f4dad8e22b1aac2447d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef03dab723665c0d72bf7c74ff9a09cf
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c04f63db1c1618f3af76193d06ffe827(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_96648ab62794120b6b287f6a88df6618(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_67f83f43d7f0507c1ae925383882ae57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a4fe64c8b9661250e12b0b4758f081cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_642f53c1fc3375431dcff3d0dad5fbdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_487d5a607372d92b396188531d59acf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_11ef3f315c31158111fbcf8efb3d9ee5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bd9d2fc7ba78d638700e08bff96f0929(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_174e248b64165ee5256e4b5630285a5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2cb9e104b84d3992f61acb6660423795(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_84bfa97ddca9cc1a027197e6a34b0c3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4c8dd0b5998bb2c77f6295b69d6cd8c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_adef93ac984eb11a022d634106f358ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_92c79eb70b103b2f1f8f1eff7a6d0adf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1c6ae52c0157db83932a8d69d1c57b46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 16, 16], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5f757dd15cbe049844ac7df1d5845571(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 16, 16], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cdefd9f3dd880f960c53ca8bb995455a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 16, 16], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f90994baf9b07e8ce040ca1bd43aa321(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 16, 16], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6b0404d68cd34462896c131635f14c4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 8, 8], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_63ae0ce4dfa52820f5596b93b469dff2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 8, 8], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b08e19324c481ffb12f9ce87c66d6aac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 8, 8], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_aba814ca1b89b9d7485209dd1732da01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 8, 8], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0197d3cd4979f7fd15c9163b3fe972dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5e0c85be79ffe9db6eb03b52742b1b31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d1a09490eb67688156e34b9b27772e33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2100, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d024cbb6d1020a84ed3ad661f8fa5f2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2100, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d97943af06a4a05357536ef233b38584(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2100, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b7381e912b8da77a017fc4f43107d26e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_71f7b56204770c5bfd3ee168a010d663(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8c79273ee640961e17e73c6522ae34b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_03e8168d8363f28ed1dda2b9a722b2a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2da7ae3c8d29b3ed7515fe0ddc9e1505(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_03e8168d8363f28ed1dda2b9a722b2a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2da7ae3c8d29b3ed7515fe0ddc9e1505(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_03e8168d8363f28ed1dda2b9a722b2a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2da7ae3c8d29b3ed7515fe0ddc9e1505(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_abb86a3d7442f3824fc647d1d1f28925(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2048, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_eec7b61bdd92d9e63624b7195b1641c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2048, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e033907e4ff9e796fc484c908f198c86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_808fd7d99bce409028234f533aa1378a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_68292c1f37c3411379b185141269590a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 14, 14, 384], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d44ae8f132fb111aada6568f89f2a0ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 15, 25], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e331210edbafec983c243879da12104e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 15, 25], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5d052416665deb50b19f8ab15be0e147(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_57b66bb5d1da58518b74ed042ab8d43c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bb9c14bc05258f185ac0ca230cb3db62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 14, 14, 384], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_58e670ca36f569ebef9c234b52a3011d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc5cccfcac4003e23fe5e6e7c450b89e
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9a18c96785da9dd8790eda55fa70f746(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af389d70137896b99943816cd6c674ce
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_663645fbe3860353b293beea7f7ed8aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef03dab723665c0d72bf7c74ff9a09cf
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f29946182154e2d98e2e7a4fccb14f20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc5cccfcac4003e23fe5e6e7c450b89e
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_500178334a6354128e98376412b2212b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af389d70137896b99943816cd6c674ce
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4743590b18a4810171412d987aa7714c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef03dab723665c0d72bf7c74ff9a09cf
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_58e670ca36f569ebef9c234b52a3011d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc5cccfcac4003e23fe5e6e7c450b89e
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9a18c96785da9dd8790eda55fa70f746(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af389d70137896b99943816cd6c674ce
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_663645fbe3860353b293beea7f7ed8aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef03dab723665c0d72bf7c74ff9a09cf
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_79db74fc1c465f46928ca89379482d24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1025, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f207ba317a8f2d1894161529c5fb0467(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1025, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d524d29271f2019a7bd1416f9fe36d4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 1025, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f0b59f90f63263e5030bebca8f866441(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 1025, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_43c3aac9a583f47a6be5ac374e161169(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6e81743cb1d0700dfd04ea418e38590
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 1025, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bc4c9281520874ed7a8efde7b9eb1b14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([44, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bd38498870955c52a966156dcc68fc30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([44, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8ff605a9ae02a03a2ca9001e4957cb7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([44, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_21c829751781bfb74a8bc6a193976e47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([44, 8, 288, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f29946182154e2d98e2e7a4fccb14f20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc5cccfcac4003e23fe5e6e7c450b89e
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_500178334a6354128e98376412b2212b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af389d70137896b99943816cd6c674ce
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4743590b18a4810171412d987aa7714c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef03dab723665c0d72bf7c74ff9a09cf
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b5a01ae21878130bc19cbf5dbf421cb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 80, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ec8465c70093722bb68e3e099d408d9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 80, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_838862a1e65c4b61919086e729ba8699(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 80, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9667323710fb68506a8df5e8a84b1e0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 80, 14, 14], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_53f2d292b8f704e76509406420c063ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2c4fe2e3b8be465edffea9f38bccf4ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a2ffa2b28e11ef87d2077af5a2de7252(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 56, 56, 96], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c60482f635ddad4d292b655a3bf4f2a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8318d1ff1b398b6f29c70db7fd5d5774
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b7381e912b8da77a017fc4f43107d26e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0f5cb0df636a5c41f366a790911a541e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_71f7b56204770c5bfd3ee168a010d663(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8c79273ee640961e17e73c6522ae34b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e7fdcf37bce1974280bdd07d3228f487(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1024, 512], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fcdb1768ef0839590a7039a4ef9a2bad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1024, 512], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_76bc650bf9ed556da4629558a7a99350(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_70e2b9988bfbc03ab0440d345f66f0f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e06717104806e0fb90152e6d6693f160(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 11109, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c2fd875a7be8b4dff02c03cd1eef0586(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 11109, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_046b688565f4d80ab983b2b96c1c5372(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 11109, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f2ec23148ff2f4e13eaf5021918c8ff3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc5cccfcac4003e23fe5e6e7c450b89e
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_324c9c6a0129d4cfd73171c180812979(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af389d70137896b99943816cd6c674ce
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b2c28bc9563d9f57cf926ffda54860fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef03dab723665c0d72bf7c74ff9a09cf
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_54a2d5ec4e6517146bcce0e77b3d3b10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 49, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fcb6ec80da738734d47c9cb8864d9283(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 49, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_19279ba78765fd5d58fb5d30a2a0bdbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 7, 7, 768], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7f3da502c070c1f3292aacd6beb30939(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc5cccfcac4003e23fe5e6e7c450b89e
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_899c745138d6a89963cd276a4a2ed280(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af389d70137896b99943816cd6c674ce
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_837eda0b4f643f4dad8e22b1aac2447d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef03dab723665c0d72bf7c74ff9a09cf
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_30121899b51bacd2490dc9e15e71c8bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 49, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_223ee0703cfed028f7dbe66d897946ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 49, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b127430db3d4e88f5aef8f05792649f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 7, 7, 768], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5c22b46430911540a8cce79bb967103d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0b99fca1e0790bb6532d75af31c50044(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 24, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7172c808fac83c18f95d7ddde6916a7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_46d9c3ec1e5c54fc5a15481689a52a75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_11ef3f315c31158111fbcf8efb3d9ee5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bd9d2fc7ba78d638700e08bff96f0929(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0915501cb4d4af3e3a0a34d13a9318d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([144, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8d661469be6719c34329ae81e20a8820(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([144, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cdac12bd403eda64d904d6275457617a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([144, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_491d0fd49617295488787aea03d8c2db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([144, 4, 96, 24], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a62489bfcf93a53901b93f48805f4b17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dda6bfecf704ff4b77311ee3d3c6b1b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3c0be17cd0a3aed862e085138ac68a67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 1024, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_41686e8fb55b375f8be3f2e559c08f55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 1024, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ff0ebf24c2811968b938165eebcf168a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 40, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d56370675a440070ed1d8de3e7fbbd11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 40, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3f2a2f2c8b87581e517b580cba32a686(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 40, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_aba20f6f4e08aa92d142842cd7b690b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 40, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e9e918737eb94fcbf51a6e5c094d5e11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 80, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3f0c2a174bf7eefb3fa97318ba337cba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 80, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1aa59bd3db723b733d96e05eb96523a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 80, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fe1f991aa0f29f4eb9a013b6ec6e672c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 80, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3aa85372d0336b9850993983a7d2fa3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 160, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3dd5a4beebcb596858368f1ac6daf871(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 160, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_96c6e01eb341b804f7ec750a88947f21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 160, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_48076449e931c1497e6c58413b18cbec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 160, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e9cc4b75e09a6a1e5118f5acec4b5140(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 384, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c154d8140e0e4e86d308841b49ffddb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 384, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d3e2b751039b40ddff50c1129e9b3aa8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 384, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6a9daa94a10c835b6cb7830b41d48e75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 384, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c260f358ce6f3ea06d3dc35aa3aeb5ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_be8f716de5992b4e8d15313f8d761bc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b46d8a72f422ec0742f9da46a876b027(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_527a17390fda568e76f26feec62626a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_935d3f871074dd33dda83278d43145ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 96, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2aab55340ed901b658de6194b1125529(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 96, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_135c004e98c31de47bc4a20f9d055d93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 96, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_24d9f0c832f838068b2de06c4a8f2702(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 96, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_994e14426e1e005b78b0391b28e826ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5fa73e1e3e9d746f9fe7bc7fe5145c01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6979ace991ac4c4fb005619b8721854a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 512], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6725c9800801ef211768da2819c370a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 512], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0e7b1bdf28fb693ae3bf3de21adda386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a974fcd1afb49f82f5208caa6fa005d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1ed3cba9d7ec326c223ad09252ab2dee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 784, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_eca6ca7eb8f980aa860dcc298e045f71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 784, 192], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6656f17b2316a05d244292949731c7c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 28, 28, 192], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_57614c9f885ba11d94c72575661017cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1024, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f3134d25c9c255bc427900eb0278e350(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1024, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_98e691a2da41d304c1de705e4d56d986(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_43193033202c1a35556f209ac7b4e112(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 6, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_082b6cf5370884e2c949bbe77392d164(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 232, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_becb158eb8cea103358ef23eef3f428e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 232, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d2cec7f2565fdf5f5d0b8951fa625de2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 232, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_552ea435864ed71dd6d331e13bda266d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 232, 32, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b360c1d6c26750e19dea7642ad74c199(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_812cb804a39a3a25910e4419767aa522(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 12, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_065240a26ef85eb4ce8896b994361d28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_03ef8d86bfce57aac2f9c110ba333614(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d387262ed8a5e267f90da4b30e0b9487(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 56, 56, 96], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0ac13fb1b776e266c6a36d935d3bb869(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 40, 28, 28], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b0f5376dc5ba086e657c0a203140503f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 40, 28, 28], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4f77a21289bd800870e57ea5b2976ffa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 40, 28, 28], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_144515d5da66e181bea64db52e57f15e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 40, 28, 28], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bb09818a66d6cf09441501a4fbc2bc5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 1024, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_63cdc0516365b897af7a2e04a3860096(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 1024, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9842b448820df43a6778760b0c6179b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 512, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b6a2363ef42df6f0e1703abacad76f91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 512, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5074c13273451770e573e18e65b767c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 97, 97], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9a4e55da8a59a30950a4fd5720db5b56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 97, 97], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9173b1e9650dd8f1af796b5841fed335(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 512, 8, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_29379a2363591b56ffe579689e89666c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 512, 8, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_660080da8fd3619c2e9d64ac26b7750e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 512, 8, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1e5b381b56e0fa8a9e4fa56488b0e33b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([3, 86, 3, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_05d9ac7ca9e5fe01c711ede49267e23b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([3, 86, 3, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0612211ee6a3fc89065761e8ab10755e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6e81743cb1d0700dfd04ea418e38590
    def get_inputs(self):
        return [
            paddle.uniform([3, 86, 3, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_02d76db4efe1cb09af3b78031bb05d11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 512, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47b87ff5068c27652f7a1f3ae7be3638(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 512, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_03e8168d8363f28ed1dda2b9a722b2a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2da7ae3c8d29b3ed7515fe0ddc9e1505(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6cc277dcdabfbda3e8fa5c3e0d31e75e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9c3a0dae9c7784964d0f682058fbf242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a9c47ee42edf8efa2f7e5eb581c120c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 512, 4, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a720a41665cf6e02452b7f7c6e9967f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 512, 4, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a3ddb35024d7d17b13fe960d51fce6e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 512, 4, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e033907e4ff9e796fc484c908f198c86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_808fd7d99bce409028234f533aa1378a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 196, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_68292c1f37c3411379b185141269590a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 14, 14, 384], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8b4f5476b614a727a5b1458e42f14389(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3024, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_758010352630481a1f26dd261c46fc25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3024, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a609127dd54b5c8920d5020833f361e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3024, 68], dtype='int32').reshape([3]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e6e9013b5d501807acbb1d6c68e06a7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ee20b24d0dc5c0f8dfc8a85b251f263
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4e2711c137e4bf01b13fe9ef32feeeba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab25d3d5c606d444873ae972ad49a88d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8d8ecd7b2dfa96750ff07e35478d95ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_206bb51737a40e49bcfd02c33c6f47a8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_603c8987a69adb22acecf007118495b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60c3a53dc70160884ae36824e2bb7ab9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_40cfe8d0af1886ba8ce5801b197c24da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ea652a9b28ef535190e472ee073bfd6
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9dbe0cd9fbea8f6a1a2d32cc29758e5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc217cebb3119779c263c0b26c1c0e92
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_407e64e76f6d09805996ede577f0a2f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65b631facec9038f7274627dd8b3d34b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_857b32325a5962b03069a1f80a346918(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1abc5a84c7b67e7998017479b702593
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dec3fa402e9ad15d50adf543e96c6768(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80551ea934c2d61c03fa6457d9ce1985
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ba4ccc45c03a4479263b04a5b04a8b4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7dfb4ada84d78f903e5538507d3b7ab
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9e4a2351220f4ee0b1040f944892f468(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91e429e4f0dfc5c2769ccfdc4607b6fa
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1761b93f00607a8cda37254fb255af33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1262774c9edc7ff8f4ba3df5056d811
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cc8a15fe980b488271eee75090bf9029(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b98c0cd022d0194b4c0832b0db60eac8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_399691f3b47d8ff7df89858e1378eabb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aaac1a3979662b89e725d317a548ca91
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1c245e8c00dfa90f04c5d339879f2818(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d52063c93809e6ba0fe1e83fac1cc5b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_474074a89caa078f8fcff9f196dff248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_33218763cd1a25505d53cd5b29035e0a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_de9381e4d08a06bcf00d693eb5d9d715(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_786c42a7e462a07279f1e57ddde082e0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
            paddle.to_tensor([17], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d483ca1c78da890226d62fa7f25fe164(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f768750d0804989fa348867c53b0bb1
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([17], dtype='int64').reshape([1]),
            paddle.to_tensor([18], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c5cbf0201ca960d94de400f794bf9d5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aab8b254011b8ce61407c27f0f4c642f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([18], dtype='int64').reshape([1]),
            paddle.to_tensor([19], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5443f7c915aa49422e723faaa19c9c48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adacbc4e820398794baec64fd98a3eb5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([19], dtype='int64').reshape([1]),
            paddle.to_tensor([20], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2312777daf9e0c7ba1d50bb35b472d40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c129e38203897bdcdc39d76ab610abe6
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([20], dtype='int64').reshape([1]),
            paddle.to_tensor([21], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b199f78770cba04512f4d1541128746d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60c299b11353f17bec6c5760b2ec71e6
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([21], dtype='int64').reshape([1]),
            paddle.to_tensor([22], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e69faca1f7585567be20b3062844aca4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd1b9dfb33b0bf8fc2956226a617717c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([22], dtype='int64').reshape([1]),
            paddle.to_tensor([23], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5df366f4f3941bff4134976627f0b117(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f598268b6c5d69f0bfc5945c4ef367f5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([23], dtype='int64').reshape([1]),
            paddle.to_tensor([24], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_42b79532dcecb21c192579fc3bfca5f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fc7f50d72d3a27bf2d77c6cfb11617a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([24], dtype='int64').reshape([1]),
            paddle.to_tensor([25], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8f25711598260759a697d7f9a08a59d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97dfa6384f0e00834b4731f732250d7c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([25], dtype='int64').reshape([1]),
            paddle.to_tensor([26], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4d010ba4d317f2e12916ac8dc4d06c5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86b557c5989b47fdac5f937cb60ef377
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([26], dtype='int64').reshape([1]),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9ba11731a312888a58303947a41bd2c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf328c3e1784a95d1c8c5174bbfc5ba
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
            paddle.to_tensor([28], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f5d5d0a11c2355336bf7874d7b722eb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b66c62fc3e483a566edf5e8f981ea3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([28], dtype='int64').reshape([1]),
            paddle.to_tensor([29], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b1ddb681a3ea43cb484798946243117c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4ba03ff4391abf9fd557ef65dd6325e
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([29], dtype='int64').reshape([1]),
            paddle.to_tensor([30], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_78b0390d72540cfa3225080960c944b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9f54fa33b7dff41838d02ed1f33cc2
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([30], dtype='int64').reshape([1]),
            paddle.to_tensor([31], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8d402d78519ab38a33cffc5d1e6dbc08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d35009155b9636cfbd28ba3893f672b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([31], dtype='int64').reshape([1]),
            paddle.to_tensor([32], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_798ff0eedae853dab51cf5e78f95b4e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5a686f212739ed486f6ca0e7a1c731d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([32], dtype='int64').reshape([1]),
            paddle.to_tensor([33], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_35b717834285cb68311077dd6ff2abca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7edebd0037dd4b4d9b8bb90546382d1a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([33], dtype='int64').reshape([1]),
            paddle.to_tensor([34], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a1e70e85c5a23406d626e034ae48944d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c769d7c7b73b95aa762264fb20471cae
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([34], dtype='int64').reshape([1]),
            paddle.to_tensor([35], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d8296745b15787d50d56cb71101d2917(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1db492ab74727de620b8a48a35c176b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([35], dtype='int64').reshape([1]),
            paddle.to_tensor([36], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cd4a57e99417d71ee9022d4d4f81ab1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43d5b23888ac8f77827eaf5ea24e488c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([36], dtype='int64').reshape([1]),
            paddle.to_tensor([37], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a8b777ad16b7c40e0327fada0b98a36c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf0e593fb3d8c20762680d907541d9ca
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([37], dtype='int64').reshape([1]),
            paddle.to_tensor([38], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e6af1f849cadb34a047af3423c97dbcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8f3e6a0312f63512f7e28c470bd5b42
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([38], dtype='int64').reshape([1]),
            paddle.to_tensor([39], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_38f532d7063fa20523e426f3df595f72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6cb44aeab8767351869c59ba5c722f9c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([39], dtype='int64').reshape([1]),
            paddle.to_tensor([40], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1bd17e29f17cfe0cd051fdd20d1003d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55e97202eb36c293cb8f7f714219eadc
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([40], dtype='int64').reshape([1]),
            paddle.to_tensor([41], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7187f316ddd29ef25da82ef58c151d77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2fe4027be3f35625d7febc50a58d4429
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([41], dtype='int64').reshape([1]),
            paddle.to_tensor([42], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b7d688707358b14817d4783564107261(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa07965657440e2272b2339a117b9b51
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([42], dtype='int64').reshape([1]),
            paddle.to_tensor([43], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_07e86f993646067ba7c94f7b77bfe23b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c25c4e732671e4d40e22bb7b9fd7c26
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([43], dtype='int64').reshape([1]),
            paddle.to_tensor([44], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1f15eb719d2f57eef2aa56852fcb8156(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e94d2aba926c9fa13ac0e1831830c67
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([44], dtype='int64').reshape([1]),
            paddle.to_tensor([45], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5cb734d8250ccea62de98ae15955bb62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a31b0d0b621cb336425463df47f5a24
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([45], dtype='int64').reshape([1]),
            paddle.to_tensor([46], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c2318fc7329202a8d901e49450c6ffea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d37c28e6918cbf046fa5642b8cd12d8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([46], dtype='int64').reshape([1]),
            paddle.to_tensor([47], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5ca552df3821635c5c2f29e35d78ff66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c334926e6aca226b4a3efe04f7c96842
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([47], dtype='int64').reshape([1]),
            paddle.to_tensor([48], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fee8354dc1b36dc37207c4194aa2b61e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_990862c82a422dc21fe38251ec184b68
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([48], dtype='int64').reshape([1]),
            paddle.to_tensor([49], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4e0d32a96dbb45a0113a53b5ed6700e9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [49]
        input_2 = [50]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1c2df311f5c8e7706e5c4205d85892f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0d32a96dbb45a0113a53b5ed6700e9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([49], dtype='int64').reshape([1]),
            paddle.to_tensor([50], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_5a1a2d05820d66ffca727cc29d4ec4f8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [50]
        input_2 = [51]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bdcea10c0af2caa7a377cc3760e9732d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a1a2d05820d66ffca727cc29d4ec4f8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([50], dtype='int64').reshape([1]),
            paddle.to_tensor([51], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_5c69ec97e4a67258ac01adad33bb9952(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [51]
        input_2 = [52]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d6ead72e3c98113c730b5d0f2252e053(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c69ec97e4a67258ac01adad33bb9952
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([51], dtype='int64').reshape([1]),
            paddle.to_tensor([52], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_5d80748d29a4f94e7cb5e6f6d21d8790(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [52]
        input_2 = [53]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_31e293f8aaa8d3c0e58e7dd0be992f54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d80748d29a4f94e7cb5e6f6d21d8790
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([52], dtype='int64').reshape([1]),
            paddle.to_tensor([53], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_80c108ce744a7b69fe1586736a7e1aef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [53]
        input_2 = [54]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f8a1b043d64d1dcf439798cd7b8bac75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80c108ce744a7b69fe1586736a7e1aef
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([53], dtype='int64').reshape([1]),
            paddle.to_tensor([54], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_3ef73fd4af55ab7b6b4db8553a2dfac5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [54]
        input_2 = [55]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b1bbe0911c5dc2fb9437c64b3b795980(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ef73fd4af55ab7b6b4db8553a2dfac5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([54], dtype='int64').reshape([1]),
            paddle.to_tensor([55], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_9db32b3dec3eeb23d76a3530590f95d1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [55]
        input_2 = [56]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6b41c4d5db8124bcdb4e7f296eef3f08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9db32b3dec3eeb23d76a3530590f95d1
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([55], dtype='int64').reshape([1]),
            paddle.to_tensor([56], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_2e786ac9b71b9e446e24d68bf848ae1f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [56]
        input_2 = [57]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_696dd05cfa66f00e6c96a67d925da306(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2e786ac9b71b9e446e24d68bf848ae1f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([56], dtype='int64').reshape([1]),
            paddle.to_tensor([57], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_7b96a53d91cfbd161443176121eb5224(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [57]
        input_2 = [58]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3a0571221051a9d9440a7fdef1f60e35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b96a53d91cfbd161443176121eb5224
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([57], dtype='int64').reshape([1]),
            paddle.to_tensor([58], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_1043753952d3fa959f0068845867aab8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [58]
        input_2 = [59]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f74499f42a1f70c92bca24076673cb5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1043753952d3fa959f0068845867aab8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([58], dtype='int64').reshape([1]),
            paddle.to_tensor([59], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_05feccfb17de306c41383ed293f857ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [59]
        input_2 = [60]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e71927a772ad7249cae8e479d6cbd7f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05feccfb17de306c41383ed293f857ee
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([59], dtype='int64').reshape([1]),
            paddle.to_tensor([60], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_b0ba8e90825040391055276d4e210419(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [60]
        input_2 = [61]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5d069f3f1b4c5d71a327356fbe04e429(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b0ba8e90825040391055276d4e210419
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([60], dtype='int64').reshape([1]),
            paddle.to_tensor([61], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_0d8fd6a8a86dd0375454cf1d7167626c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [61]
        input_2 = [62]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d306353368dec2772791ddc6e314e211(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d8fd6a8a86dd0375454cf1d7167626c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([61], dtype='int64').reshape([1]),
            paddle.to_tensor([62], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_e895c4f6ec195542e035fbd0baa892b9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [62]
        input_2 = [63]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_94a39c05278aa7b77da7609d567529fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e895c4f6ec195542e035fbd0baa892b9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([62], dtype='int64').reshape([1]),
            paddle.to_tensor([63], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_1c1b0cca2c93bf0066f02e6d4c88e0d0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [63]
        input_2 = [64]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8ae6dd89915bb8f849d5e31049f676b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c1b0cca2c93bf0066f02e6d4c88e0d0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([63], dtype='int64').reshape([1]),
            paddle.to_tensor([64], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c5acabfc0720d491637bbb773b80ada2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [64]
        input_2 = [65]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aaba5b03935e7ee944e61ef440a40740(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5acabfc0720d491637bbb773b80ada2
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([64], dtype='int64').reshape([1]),
            paddle.to_tensor([65], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_aee403cf30ce9cf89e40e9010385c18f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [65]
        input_2 = [66]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b46468cc7ed994021a166f4b1c1ab4f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aee403cf30ce9cf89e40e9010385c18f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([65], dtype='int64').reshape([1]),
            paddle.to_tensor([66], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_74d47b33e66bafaec009344c2f3cbfc2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [66]
        input_2 = [67]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_61bf14f1d653b51276f965bc0d40ab01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74d47b33e66bafaec009344c2f3cbfc2
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([66], dtype='int64').reshape([1]),
            paddle.to_tensor([67], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_57b0c95cca5d3515e53e02ab0eb66488(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [67]
        input_2 = [68]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0e8807ea88c18dea559a03994ba2338e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57b0c95cca5d3515e53e02ab0eb66488
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([67], dtype='int64').reshape([1]),
            paddle.to_tensor([68], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_1e5f997e02fa83447ecd2bc79dd08e54(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [68]
        input_2 = [69]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2feed57f163b0e78516617df0a406bfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e5f997e02fa83447ecd2bc79dd08e54
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([68], dtype='int64').reshape([1]),
            paddle.to_tensor([69], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_7991a7d928744aae50d38b43398fd241(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [69]
        input_2 = [70]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6f184ebac41266fe1a0c980b169e2a6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7991a7d928744aae50d38b43398fd241
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([69], dtype='int64').reshape([1]),
            paddle.to_tensor([70], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ea6e775a07186afa02579253466db757(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [70]
        input_2 = [71]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2ab1d93575a38299d7d95f3c861f038c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea6e775a07186afa02579253466db757
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([70], dtype='int64').reshape([1]),
            paddle.to_tensor([71], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4ff1a9bee13250132d6c5d8ea023564e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [71]
        input_2 = [72]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_31ef5029806d0f6e8ee9598890a7e554(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ff1a9bee13250132d6c5d8ea023564e
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([71], dtype='int64').reshape([1]),
            paddle.to_tensor([72], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_2390ac7af24a9da823f4db904a27fe01(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [72]
        input_2 = [73]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dc4b3d106c93eb14ab52c5323eda0ddf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2390ac7af24a9da823f4db904a27fe01
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([72], dtype='int64').reshape([1]),
            paddle.to_tensor([73], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_3049843ec8c6f5da4acc36edfe875866(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [73]
        input_2 = [74]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_562a67366f00521bc9c986685006a760(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3049843ec8c6f5da4acc36edfe875866
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([73], dtype='int64').reshape([1]),
            paddle.to_tensor([74], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4f8a18fea6afaba2482d07fc0e7889d4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [74]
        input_2 = [75]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_18f9148837723d75aa450cab388f8545(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f8a18fea6afaba2482d07fc0e7889d4
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([74], dtype='int64').reshape([1]),
            paddle.to_tensor([75], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_e7e3f65748080ddfd7240a6ef3d39979(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [75]
        input_2 = [76]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_af50cc5d009c93efdc3fed26e347bbac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7e3f65748080ddfd7240a6ef3d39979
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([75], dtype='int64').reshape([1]),
            paddle.to_tensor([76], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_96f8b6da552bfc7876b637f165be3097(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [76]
        input_2 = [77]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d04e07e098584653665c846fa48dd669(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96f8b6da552bfc7876b637f165be3097
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([76], dtype='int64').reshape([1]),
            paddle.to_tensor([77], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_9239083d18913f353726ca52762f7fd6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [77]
        input_2 = [78]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4e016398f1567e47350b3a14b5c29828(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9239083d18913f353726ca52762f7fd6
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([77], dtype='int64').reshape([1]),
            paddle.to_tensor([78], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4b3960dd074137c221b8f0e6e04d1620(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [78]
        input_2 = [79]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_089a594d3a7b97daa4d1be939d23fcef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b3960dd074137c221b8f0e6e04d1620
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([78], dtype='int64').reshape([1]),
            paddle.to_tensor([79], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_66ae5987bc4a1642992678b9929abbae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [79]
        input_2 = [80]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_77cac0b317bf9837152937c25ffdb147(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66ae5987bc4a1642992678b9929abbae
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([79], dtype='int64').reshape([1]),
            paddle.to_tensor([80], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_a1e602063ed643966b53122ed1f2de93(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [80]
        input_2 = [81]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_71ab16f78b80a9776de7f0440384348a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1e602063ed643966b53122ed1f2de93
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([80], dtype='int64').reshape([1]),
            paddle.to_tensor([81], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_1a95fda0d0e1f0d641f4438e3e27d2a1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [81]
        input_2 = [82]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_752c07337890a1283aa4fbc299acc1c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a95fda0d0e1f0d641f4438e3e27d2a1
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([81], dtype='int64').reshape([1]),
            paddle.to_tensor([82], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d8c805b590dae688b396fa5fb15c6855(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [82]
        input_2 = [83]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b8204949e53091fe2d4e71d7fc4cae07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8c805b590dae688b396fa5fb15c6855
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([82], dtype='int64').reshape([1]),
            paddle.to_tensor([83], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_b6291d1a8a909f6c721f4c30e59ca5e6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [83]
        input_2 = [84]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_316c9a4d30d3763819edeaf9b0d180a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6291d1a8a909f6c721f4c30e59ca5e6
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([83], dtype='int64').reshape([1]),
            paddle.to_tensor([84], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_3d5fa331923d25924ac5ef34ed774c2b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [84]
        input_2 = [85]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d46228d418224242d6b7a1fee9ee7eb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d5fa331923d25924ac5ef34ed774c2b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([84], dtype='int64').reshape([1]),
            paddle.to_tensor([85], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_52a05474501687d3e67aacafa1d1f041(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [85]
        input_2 = [86]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5fd3096096acb990f0d9ec338f5804eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_52a05474501687d3e67aacafa1d1f041
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([85], dtype='int64').reshape([1]),
            paddle.to_tensor([86], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6baf0ea912ad571c5a2a0276d48c2219(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [86]
        input_2 = [87]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dcdff5bed515231fea0dea6a3c6143ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6baf0ea912ad571c5a2a0276d48c2219
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([86], dtype='int64').reshape([1]),
            paddle.to_tensor([87], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_86bc90ecfc7444548107017ec1af65f5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [87]
        input_2 = [88]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cfba1c3399d2fa29f575475398bae7f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86bc90ecfc7444548107017ec1af65f5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([87], dtype='int64').reshape([1]),
            paddle.to_tensor([88], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_bc609ae85b43724b9bec4ff4e8eb6f1d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [88]
        input_2 = [89]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dd22129308ddd4915451d7214ec3499f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc609ae85b43724b9bec4ff4e8eb6f1d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([88], dtype='int64').reshape([1]),
            paddle.to_tensor([89], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_1e0c91e422ecf690fe886517d61b8293(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [89]
        input_2 = [90]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f2a1f3cffbe7726320a2b005a25a72b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e0c91e422ecf690fe886517d61b8293
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([89], dtype='int64').reshape([1]),
            paddle.to_tensor([90], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_e61de5b9a65a2d3af85eb2937038fa68(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [90]
        input_2 = [91]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e32607a484100d0706f25ab386f425bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e61de5b9a65a2d3af85eb2937038fa68
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([90], dtype='int64').reshape([1]),
            paddle.to_tensor([91], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_34c71ee3f4f873cbbe7f4290a1e39442(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [91]
        input_2 = [92]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_20734730c1002c05fd70484c1c5d7bdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34c71ee3f4f873cbbe7f4290a1e39442
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([91], dtype='int64').reshape([1]),
            paddle.to_tensor([92], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d059d662e50acc6d6a865d82d6a93c65(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [92]
        input_2 = [93]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_90970a46141cffa53b745ba50ae5eb11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d059d662e50acc6d6a865d82d6a93c65
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([92], dtype='int64').reshape([1]),
            paddle.to_tensor([93], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_365b4cb8b07a13fc30cd71cf0fe54467(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [93]
        input_2 = [94]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_774ed35786a276c4104189f014954fd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_365b4cb8b07a13fc30cd71cf0fe54467
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([93], dtype='int64').reshape([1]),
            paddle.to_tensor([94], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f2d6dfc75552c08d0984a385f6125ac8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [94]
        input_2 = [95]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d82e668f3b8e92d61a5f75c29940e56a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f2d6dfc75552c08d0984a385f6125ac8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([94], dtype='int64').reshape([1]),
            paddle.to_tensor([95], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_60372a23b75cfe7fa257ceb23a08b78e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [95]
        input_2 = [96]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_df7b997646e26a48325585db5f1a1cca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60372a23b75cfe7fa257ceb23a08b78e
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([95], dtype='int64').reshape([1]),
            paddle.to_tensor([96], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_31b0c27a5b43e92409de9240e89c971d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [96]
        input_2 = [97]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cb96f1eaaf1107165ea4a8cd112b9221(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31b0c27a5b43e92409de9240e89c971d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([96], dtype='int64').reshape([1]),
            paddle.to_tensor([97], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_9a8fb6b9a20077738a25d5f5c1616e29(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [97]
        input_2 = [98]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3e3bd3510b5132ca4121e04d4369ff47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a8fb6b9a20077738a25d5f5c1616e29
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([97], dtype='int64').reshape([1]),
            paddle.to_tensor([98], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6df01adaefb3145d2d0df98046fbb7d7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [98]
        input_2 = [99]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eadccbfced4501fcf392fbef7f76e267(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6df01adaefb3145d2d0df98046fbb7d7
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([98], dtype='int64').reshape([1]),
            paddle.to_tensor([99], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_e6f445e23bd8dc4d3ecefbe70578745b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [99]
        input_2 = [100]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8129aabd3e3803f30a9b2f2053bb59b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6f445e23bd8dc4d3ecefbe70578745b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([99], dtype='int64').reshape([1]),
            paddle.to_tensor([100], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_1a8e28e35c39b39d488a146270d01f2a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [100]
        input_2 = [101]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dc8d38be33babd8dc670f46a50363e62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a8e28e35c39b39d488a146270d01f2a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([100], dtype='int64').reshape([1]),
            paddle.to_tensor([101], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_e4b2750e219d2906b82e83560f9621c8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [101]
        input_2 = [102]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9172cceec1f9fa00da6b8693aeaa6015(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4b2750e219d2906b82e83560f9621c8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([101], dtype='int64').reshape([1]),
            paddle.to_tensor([102], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_44000737be7d093efdd335595a516185(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [102]
        input_2 = [103]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_17f6ffd28f6a1ccda04929e09f57b820(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44000737be7d093efdd335595a516185
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([102], dtype='int64').reshape([1]),
            paddle.to_tensor([103], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_41e8f10da0b335996425cf90621610c3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [103]
        input_2 = [104]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b5705d3e16ea8a326e06fd49ad6d3919(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41e8f10da0b335996425cf90621610c3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([103], dtype='int64').reshape([1]),
            paddle.to_tensor([104], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_305db70aac390300c1be27e558d0bc5b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [104]
        input_2 = [105]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0cc03ab4b197b0b1e5fc4cec24642d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_305db70aac390300c1be27e558d0bc5b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([104], dtype='int64').reshape([1]),
            paddle.to_tensor([105], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_b231fb2c6b47184e0ec9a6c1789788b2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [105]
        input_2 = [106]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5f8859e4040284cb61e819caf6d9f8ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b231fb2c6b47184e0ec9a6c1789788b2
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([105], dtype='int64').reshape([1]),
            paddle.to_tensor([106], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_3e6527cee8dc3e92fadd966259616e81(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [106]
        input_2 = [107]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_211e6d55b7baa60c74725c8e0018b9f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e6527cee8dc3e92fadd966259616e81
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([106], dtype='int64').reshape([1]),
            paddle.to_tensor([107], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_cca97692cbd4619491ca3eeed121c924(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [107]
        input_2 = [108]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1608349491d0de81cd07bcd412e11337(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cca97692cbd4619491ca3eeed121c924
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([107], dtype='int64').reshape([1]),
            paddle.to_tensor([108], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_5efd3b93f1979f2e262d9c0a80b606a9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [108]
        input_2 = [109]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ee3e2e130a3d13ad53c51d446b77240e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5efd3b93f1979f2e262d9c0a80b606a9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([108], dtype='int64').reshape([1]),
            paddle.to_tensor([109], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_abb17fd2bb7b9f473fe1166805bbf615(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [109]
        input_2 = [110]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_44de2b7a3b5cd0864d9c6c8346c23b55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_abb17fd2bb7b9f473fe1166805bbf615
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([109], dtype='int64').reshape([1]),
            paddle.to_tensor([110], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_624830c9200922b19550dc209a00afba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [110]
        input_2 = [111]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_41f6ebccbf31bc10f5b0b786ed5abae7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_624830c9200922b19550dc209a00afba
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([110], dtype='int64').reshape([1]),
            paddle.to_tensor([111], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_3c342036a1d409451346983e91ccd875(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [111]
        input_2 = [112]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1bdd68b595a87179bf349d9c1d9616e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c342036a1d409451346983e91ccd875
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([111], dtype='int64').reshape([1]),
            paddle.to_tensor([112], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_eff7eb9381137e267e71dc30efbdf424(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [112]
        input_2 = [113]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_04f3f73ba806503ea6935a4a7bc0aa37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eff7eb9381137e267e71dc30efbdf424
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([112], dtype='int64').reshape([1]),
            paddle.to_tensor([113], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_b5add59e2a9581b77a2032ee455b578b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [113]
        input_2 = [114]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c1ef67bf9a105805f275f0a8ce0c829a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b5add59e2a9581b77a2032ee455b578b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([113], dtype='int64').reshape([1]),
            paddle.to_tensor([114], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d7e2c20ffc4e6a5bde11771b655d1c1b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [114]
        input_2 = [115]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2330228b0d00b7cdd5eac44f0d0a11d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7e2c20ffc4e6a5bde11771b655d1c1b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([114], dtype='int64').reshape([1]),
            paddle.to_tensor([115], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6f1195eb2236a6cd4484a1c5d1e18489(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [115]
        input_2 = [116]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ee1aa6b67425ed5c2483ddf11759471e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1195eb2236a6cd4484a1c5d1e18489
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([115], dtype='int64').reshape([1]),
            paddle.to_tensor([116], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6e8c720839779bf389fb6974f502cc1b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [116]
        input_2 = [117]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c876bcfdc8bbf9d795c254239bb95bc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e8c720839779bf389fb6974f502cc1b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([116], dtype='int64').reshape([1]),
            paddle.to_tensor([117], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_303b6958e51df417f954f2b4daaf6c8a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [117]
        input_2 = [118]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_335f2d31359abdb0d3f2c7d3defc65f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_303b6958e51df417f954f2b4daaf6c8a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([117], dtype='int64').reshape([1]),
            paddle.to_tensor([118], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f69f231e08a1a41edd3e54936dc4950b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [118]
        input_2 = [119]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8b49a51c362761570bba8165905ecf05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f69f231e08a1a41edd3e54936dc4950b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([118], dtype='int64').reshape([1]),
            paddle.to_tensor([119], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_a5b2438bb0cbd73f46e1d1dcb31b5e5f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [119]
        input_2 = [120]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_39856f819c739f36a969361ae9a45e8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5b2438bb0cbd73f46e1d1dcb31b5e5f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([119], dtype='int64').reshape([1]),
            paddle.to_tensor([120], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_b2bf1a64e260d3b461c6260b1ebb75db(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [120]
        input_2 = [121]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c5fe623b8ae02ad1ffb4b2e239d13c8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2bf1a64e260d3b461c6260b1ebb75db
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([120], dtype='int64').reshape([1]),
            paddle.to_tensor([121], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_035ba0f8e5ab1d4de8fd65825fe31b65(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [121]
        input_2 = [122]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_07c1e734bff74a00f64a02a07ceb6a90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_035ba0f8e5ab1d4de8fd65825fe31b65
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([121], dtype='int64').reshape([1]),
            paddle.to_tensor([122], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d52d735648419af21967619cdad9c6d0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [122]
        input_2 = [123]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_77b53df919c0d0915d49b33a9401dd66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d52d735648419af21967619cdad9c6d0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([122], dtype='int64').reshape([1]),
            paddle.to_tensor([123], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_e889360d6ffa93d8e223cd0ae4866b23(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [123]
        input_2 = [124]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1e32d26d704e15722b561e74e7735634(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e889360d6ffa93d8e223cd0ae4866b23
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([123], dtype='int64').reshape([1]),
            paddle.to_tensor([124], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4540a623ec8ae0a09c447b16f5046ab4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [124]
        input_2 = [125]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1c89a642fbf5d7dd73611d35969939ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4540a623ec8ae0a09c447b16f5046ab4
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([124], dtype='int64').reshape([1]),
            paddle.to_tensor([125], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6268ecb6b03d79ac6c27223bc7e3e615(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [125]
        input_2 = [126]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ddcd5ac5c3bbe9fdfb306420bea9ccb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6268ecb6b03d79ac6c27223bc7e3e615
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([125], dtype='int64').reshape([1]),
            paddle.to_tensor([126], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_063ff00d9655616bba5df4a5ff4ce389(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [126]
        input_2 = [127]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_af2360370d95a1b404fe27f0fa13e156(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_063ff00d9655616bba5df4a5ff4ce389
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([126], dtype='int64').reshape([1]),
            paddle.to_tensor([127], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_3fe7f4e9ff9cf20121ad9b6f9293a19b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [127]
        input_2 = [128]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f36a31a01b1c1cebe3990d5f40667762(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3fe7f4e9ff9cf20121ad9b6f9293a19b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([127], dtype='int64').reshape([1]),
            paddle.to_tensor([128], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_1fd4c0827252aa57d4d4214d2088e824(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [128]
        input_2 = [129]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3957c5b780dae743631d82717e3c6e39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fd4c0827252aa57d4d4214d2088e824
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([128], dtype='int64').reshape([1]),
            paddle.to_tensor([129], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_91584ec03c966f7f18f22dca59197c6f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [129]
        input_2 = [130]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e7f824cd7e70a83e0f9587b317a32379(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91584ec03c966f7f18f22dca59197c6f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([129], dtype='int64').reshape([1]),
            paddle.to_tensor([130], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6891dc43ce63e78c45bdf316bced5dea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [130]
        input_2 = [131]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_07e7c364ca80532f111cf1f22a28bdf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6891dc43ce63e78c45bdf316bced5dea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([130], dtype='int64').reshape([1]),
            paddle.to_tensor([131], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c864048bbf39a01014aedc0f643910c1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [131]
        input_2 = [132]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a7eff36ab7100025f903a989f336925e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c864048bbf39a01014aedc0f643910c1
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([131], dtype='int64').reshape([1]),
            paddle.to_tensor([132], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c40e54f01b1855fd2e8770ded5b22d43(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [132]
        input_2 = [133]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_62e2f841c66570c380521db73f381681(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c40e54f01b1855fd2e8770ded5b22d43
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([132], dtype='int64').reshape([1]),
            paddle.to_tensor([133], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_a53fc68e7849caf08871010442a5ade9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [133]
        input_2 = [134]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d7ee96696165ac62de7d05170fa12f31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a53fc68e7849caf08871010442a5ade9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([133], dtype='int64').reshape([1]),
            paddle.to_tensor([134], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_bea07176d53583909a1f46ab76e6a4c7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [134]
        input_2 = [135]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5e968da4f50140fb8625882908bc5942(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bea07176d53583909a1f46ab76e6a4c7
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([134], dtype='int64').reshape([1]),
            paddle.to_tensor([135], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_e0dead4f69b3d560c055c651c66b29bc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [135]
        input_2 = [136]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0f8cf8d8ede1a26de2acb5ae41cfcacc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0dead4f69b3d560c055c651c66b29bc
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([135], dtype='int64').reshape([1]),
            paddle.to_tensor([136], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_0ef152e47d921a091d74d0d82f7a5d76(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [136]
        input_2 = [137]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bd710e5d2c3146c212ae471a3d1640d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ef152e47d921a091d74d0d82f7a5d76
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([136], dtype='int64').reshape([1]),
            paddle.to_tensor([137], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_eb56c29951f2fc23c37a40e90a3bc9d9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [137]
        input_2 = [138]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aac3cd0c00a8ed1587f64fd30e56e2ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb56c29951f2fc23c37a40e90a3bc9d9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([137], dtype='int64').reshape([1]),
            paddle.to_tensor([138], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_721f528162ec6a39d68661d2b6049dc5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [138]
        input_2 = [139]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e59294d98a3abf197e263f817daa9d5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_721f528162ec6a39d68661d2b6049dc5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([138], dtype='int64').reshape([1]),
            paddle.to_tensor([139], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_cbfadf0e9d0f8f7721da257275711a15(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [139]
        input_2 = [140]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_82e86da848ea269ffb4679451adb7337(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbfadf0e9d0f8f7721da257275711a15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([139], dtype='int64').reshape([1]),
            paddle.to_tensor([140], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_33823a2e8007081d726b6fdc3e1fd2fa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [140]
        input_2 = [141]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7f8897e8042426c232059a81248ff291(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_33823a2e8007081d726b6fdc3e1fd2fa
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([140], dtype='int64').reshape([1]),
            paddle.to_tensor([141], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_2733c6537a1074cb1ab489176b62a4fe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [141]
        input_2 = [142]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2c84f4551dd466d45995663e79091ae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2733c6537a1074cb1ab489176b62a4fe
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([141], dtype='int64').reshape([1]),
            paddle.to_tensor([142], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_dba2746a44279ed1ad3b20d2a43d1e03(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [142]
        input_2 = [143]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3097610fbc6096155ef085aeb87f029d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dba2746a44279ed1ad3b20d2a43d1e03
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([142], dtype='int64').reshape([1]),
            paddle.to_tensor([143], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c237ae9b26f60e38e327aed6c680be71(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [143]
        input_2 = [144]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c546f46f3bb7bcf5ef3c9b54b3be9cb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c237ae9b26f60e38e327aed6c680be71
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([143], dtype='int64').reshape([1]),
            paddle.to_tensor([144], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4a32e851fedae7007dc3fbf936578153(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [144]
        input_2 = [145]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6c87c53213473c36a19d7d0b3b82de50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a32e851fedae7007dc3fbf936578153
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([144], dtype='int64').reshape([1]),
            paddle.to_tensor([145], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c51692c4517ea6e565f1a77747e9908d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [145]
        input_2 = [146]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_106c43059da78204b9c4cb61af136b7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51692c4517ea6e565f1a77747e9908d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([145], dtype='int64').reshape([1]),
            paddle.to_tensor([146], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_2b44164c668d746029534332104a4a6b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [146]
        input_2 = [147]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_68beab44238bee2af1139d08fbfd1c03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b44164c668d746029534332104a4a6b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([146], dtype='int64').reshape([1]),
            paddle.to_tensor([147], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_872b988928384c492fe25fa957257996(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [147]
        input_2 = [148]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_94a9662ad5633a849047eaf13e1b88e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_872b988928384c492fe25fa957257996
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([147], dtype='int64').reshape([1]),
            paddle.to_tensor([148], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_38f631ff9d10cfee80b7a5fcd9624aee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [148]
        input_2 = [149]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c558463128dd9ee731917c45f0bfe08d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38f631ff9d10cfee80b7a5fcd9624aee
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([148], dtype='int64').reshape([1]),
            paddle.to_tensor([149], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_0eaf218a272d466e69b9fe4f8a959275(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [149]
        input_2 = [150]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d5e30f1f11fe13d36844b8390149dd8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0eaf218a272d466e69b9fe4f8a959275
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([149], dtype='int64').reshape([1]),
            paddle.to_tensor([150], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_9c21412498b5eb18fd87d291f9c7b018(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [150]
        input_2 = [151]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1bf2513cb24b566c813d2dca3ab7be5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c21412498b5eb18fd87d291f9c7b018
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([150], dtype='int64').reshape([1]),
            paddle.to_tensor([151], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_9b2b4955ea30444c4d1b3cd4e04cd7a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [151]
        input_2 = [152]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1a4b210f6ea64a252ee057777f4a58ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b2b4955ea30444c4d1b3cd4e04cd7a3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([151], dtype='int64').reshape([1]),
            paddle.to_tensor([152], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_7da1bd565cbdadeb0aed8c212b347d3a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [152]
        input_2 = [153]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_251297add0a9fcc225c4c6ee8315aba7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7da1bd565cbdadeb0aed8c212b347d3a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([152], dtype='int64').reshape([1]),
            paddle.to_tensor([153], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_1c8fe493d4ef15618410284a007894fc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [153]
        input_2 = [154]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8ecc81f1c71715d7be27b03cb97fbf01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c8fe493d4ef15618410284a007894fc
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([153], dtype='int64').reshape([1]),
            paddle.to_tensor([154], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_db75e6e8a4a6bc275301a2ec2aca0df5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [154]
        input_2 = [155]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b0c9ccae40d3d0872721fa0146fb72af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db75e6e8a4a6bc275301a2ec2aca0df5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([154], dtype='int64').reshape([1]),
            paddle.to_tensor([155], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c7b6cc8c4ce397c9aade0f0014ba7d03(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [155]
        input_2 = [156]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_116feb10ab45b9e43a19dc96e2d07879(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7b6cc8c4ce397c9aade0f0014ba7d03
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([155], dtype='int64').reshape([1]),
            paddle.to_tensor([156], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_8e40f6257413c53b62ffe24ceb3e1855(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [156]
        input_2 = [157]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_69e56e90c34a265bb955ccaff7272fae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e40f6257413c53b62ffe24ceb3e1855
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([156], dtype='int64').reshape([1]),
            paddle.to_tensor([157], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_b1a69274436c3dd69c360b58cbfc18d7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [157]
        input_2 = [158]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_01cd1542c0c31d250a4eb46d758f3251(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1a69274436c3dd69c360b58cbfc18d7
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([157], dtype='int64').reshape([1]),
            paddle.to_tensor([158], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_87e585b65627475f6bb1aaba2994b574(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [158]
        input_2 = [159]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4a9058d551da02bdd5c8a0b8436391f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87e585b65627475f6bb1aaba2994b574
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([158], dtype='int64').reshape([1]),
            paddle.to_tensor([159], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_27e6602c875c8dd5c9ca7679ea2ca3ac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [159]
        input_2 = [160]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ad97f4eb869cd28886a000ea971c9b13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27e6602c875c8dd5c9ca7679ea2ca3ac
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([159], dtype='int64').reshape([1]),
            paddle.to_tensor([160], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6a797436fd021fc878d26ee15c876950(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [160]
        input_2 = [161]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_55286abbe22464bd2d6ee69912a04999(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a797436fd021fc878d26ee15c876950
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([160], dtype='int64').reshape([1]),
            paddle.to_tensor([161], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_e0b00de88151f9a0ecfa745510f20876(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [161]
        input_2 = [162]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b3d7cec27999b9126f79bafa8ebdfc30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0b00de88151f9a0ecfa745510f20876
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([161], dtype='int64').reshape([1]),
            paddle.to_tensor([162], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_3f8deb7f7a00201ce8438bc67e0b3f1c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [162]
        input_2 = [163]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a52b4aef28277b4d0667a6ee4c5062b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f8deb7f7a00201ce8438bc67e0b3f1c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([162], dtype='int64').reshape([1]),
            paddle.to_tensor([163], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_7bb707401f90f66629ccf4d315adee95(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [163]
        input_2 = [164]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ecf90f3f155ea1ddcdf91c7c5b27d1ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7bb707401f90f66629ccf4d315adee95
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([163], dtype='int64').reshape([1]),
            paddle.to_tensor([164], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_5b564a714f2aefb69ab5a045fc378524(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [164]
        input_2 = [165]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_32a75b8ece2727c402be8f2731571acc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b564a714f2aefb69ab5a045fc378524
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([164], dtype='int64').reshape([1]),
            paddle.to_tensor([165], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_13283b7a4945b77cb4e7418384371c0d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [165]
        input_2 = [166]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_050710bb7e83bc4825760f1b19e4644d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13283b7a4945b77cb4e7418384371c0d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([165], dtype='int64').reshape([1]),
            paddle.to_tensor([166], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_a084d3690eeeb249244e9595e3f9801f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [166]
        input_2 = [167]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4d710f6384e381ab0dc4ad770f65e3ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a084d3690eeeb249244e9595e3f9801f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([166], dtype='int64').reshape([1]),
            paddle.to_tensor([167], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_69b72a3c196c21703b2dfa5b84582b4a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [167]
        input_2 = [168]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fbff251fa5d2409878fbdf4f92c517d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69b72a3c196c21703b2dfa5b84582b4a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([167], dtype='int64').reshape([1]),
            paddle.to_tensor([168], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_16d6fc542587b8d569b3d27bfac0c4b2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [168]
        input_2 = [169]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_af6639a5c608400d792cf728de2b1fdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16d6fc542587b8d569b3d27bfac0c4b2
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([168], dtype='int64').reshape([1]),
            paddle.to_tensor([169], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_47b3f8f75ea4e9987c5fbfb85da718ce(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [169]
        input_2 = [170]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_48a268f22b35ef08cacfd4349e6a6a3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b3f8f75ea4e9987c5fbfb85da718ce
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([169], dtype='int64').reshape([1]),
            paddle.to_tensor([170], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ebe071fe85545952ea97480ebf4e01bc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [170]
        input_2 = [171]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dcbddeb20ee465fbef75915eb1c300b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ebe071fe85545952ea97480ebf4e01bc
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([170], dtype='int64').reshape([1]),
            paddle.to_tensor([171], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6aad72963c1aa3bb7895ced7bbdf2485(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [171]
        input_2 = [172]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7de46e5c0615b124e5ea38dcded80f11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aad72963c1aa3bb7895ced7bbdf2485
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([171], dtype='int64').reshape([1]),
            paddle.to_tensor([172], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6dcf0ee59871e24021ce6f088962fd9c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [172]
        input_2 = [173]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7350be9df158b3ffb95363fce2e2854f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dcf0ee59871e24021ce6f088962fd9c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([172], dtype='int64').reshape([1]),
            paddle.to_tensor([173], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_309c32e5d4e3f89287180094709236f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [173]
        input_2 = [174]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_865e9f681313e6d5c8bd70ce87153ba9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_309c32e5d4e3f89287180094709236f9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([173], dtype='int64').reshape([1]),
            paddle.to_tensor([174], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4909d17f50e2dabb13ffa3c44a06118d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [174]
        input_2 = [175]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d2ae3de4438e44620b180975e9846a7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4909d17f50e2dabb13ffa3c44a06118d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([174], dtype='int64').reshape([1]),
            paddle.to_tensor([175], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_0d2b51b70c773ee954d112d2660bd288(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [175]
        input_2 = [176]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_84f4c580e0163808b9c0187241fcc4a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d2b51b70c773ee954d112d2660bd288
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([175], dtype='int64').reshape([1]),
            paddle.to_tensor([176], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_65e804ea54731364c2358848f42244e7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [176]
        input_2 = [177]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f380d65fa256884fcc669151c7712080(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65e804ea54731364c2358848f42244e7
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([176], dtype='int64').reshape([1]),
            paddle.to_tensor([177], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6e20c38159213bb07ef0e1f8b4091f15(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [177]
        input_2 = [178]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_65383a793df0ebaebbe2a014ce2d6e35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e20c38159213bb07ef0e1f8b4091f15
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([177], dtype='int64').reshape([1]),
            paddle.to_tensor([178], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_18fcc8eec60bcef645f9873df2a7986c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [178]
        input_2 = [179]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ca45d31abc50001d365564c068d75b84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_18fcc8eec60bcef645f9873df2a7986c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([178], dtype='int64').reshape([1]),
            paddle.to_tensor([179], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_469880d10575ae49c3e0631d56cb2347(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [179]
        input_2 = [180]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7750c3d13fc2632498c9e8f44188f6ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_469880d10575ae49c3e0631d56cb2347
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([179], dtype='int64').reshape([1]),
            paddle.to_tensor([180], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_85ef9b0df238489cbe326479ecc978f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [180]
        input_2 = [181]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_606c78b4e28f8951efd7f812a0f5c0fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85ef9b0df238489cbe326479ecc978f9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([180], dtype='int64').reshape([1]),
            paddle.to_tensor([181], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c7255f7e0428b52308e308a63c1c8587(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [181]
        input_2 = [182]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b7cd0385f5713cab3093167254611f8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7255f7e0428b52308e308a63c1c8587
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([181], dtype='int64').reshape([1]),
            paddle.to_tensor([182], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_b34e7c9f06c2556cfb68033f1c1a5b58(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [182]
        input_2 = [183]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4ca391897d76df558b3dca9370b5e324(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b34e7c9f06c2556cfb68033f1c1a5b58
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([182], dtype='int64').reshape([1]),
            paddle.to_tensor([183], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4defda65bffe28bfd2964d4f8cd036e5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [183]
        input_2 = [184]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_035ba9521a976910768e11f03bcc0ae7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4defda65bffe28bfd2964d4f8cd036e5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([183], dtype='int64').reshape([1]),
            paddle.to_tensor([184], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_a0d12c2d926c14ec94f1ddd6fccea713(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [184]
        input_2 = [185]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b3d23617bde7c2d81f2dbc08bfb2be34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0d12c2d926c14ec94f1ddd6fccea713
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([184], dtype='int64').reshape([1]),
            paddle.to_tensor([185], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_7e89c79fcad58cb95a62c8a47d14204d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [185]
        input_2 = [186]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b73b1092057ef78d5d15774dbc626e27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e89c79fcad58cb95a62c8a47d14204d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([185], dtype='int64').reshape([1]),
            paddle.to_tensor([186], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_3b5e7ae7858a45df76ca475cbd09ad35(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [186]
        input_2 = [187]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9f7a7c0b9d7768e40fa37f1c62ff8213(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b5e7ae7858a45df76ca475cbd09ad35
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([186], dtype='int64').reshape([1]),
            paddle.to_tensor([187], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_dbd337f10ed63caa696c60593dc35485(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [187]
        input_2 = [188]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_23a97cd52ddfa5c7307622a22a9829b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbd337f10ed63caa696c60593dc35485
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([187], dtype='int64').reshape([1]),
            paddle.to_tensor([188], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_eaf06c93b083471786a33a71ab8aa307(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [188]
        input_2 = [189]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_02c1ae8688aa09e8860f201ea121fd82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eaf06c93b083471786a33a71ab8aa307
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([188], dtype='int64').reshape([1]),
            paddle.to_tensor([189], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_9c134e61ee3746cff05f31195ae87593(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [189]
        input_2 = [190]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_11248a1751d7532ac0130982d3e9f1e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c134e61ee3746cff05f31195ae87593
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([189], dtype='int64').reshape([1]),
            paddle.to_tensor([190], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_5b2be539e8ba38560b9f2eb46042ae95(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [190]
        input_2 = [191]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d35af4d5e2d811fcfbec0e262a428581(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b2be539e8ba38560b9f2eb46042ae95
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([190], dtype='int64').reshape([1]),
            paddle.to_tensor([191], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c83b6a41fdfc4241cdc52702b1a5ad9d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [191]
        input_2 = [192]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b085928e82f9a521ca01cf72e3d8f656(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c83b6a41fdfc4241cdc52702b1a5ad9d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([191], dtype='int64').reshape([1]),
            paddle.to_tensor([192], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_234a7de7da94cdfe668e03641853280c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [192]
        input_2 = [193]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a9bc1f0723bd6ba18addddbbeb992eb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_234a7de7da94cdfe668e03641853280c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([192], dtype='int64').reshape([1]),
            paddle.to_tensor([193], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f785cbec5a5ae840b3c44d884d48ac48(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [193]
        input_2 = [194]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1e211b6479c6dacdb76b840c4c57a188(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f785cbec5a5ae840b3c44d884d48ac48
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([193], dtype='int64').reshape([1]),
            paddle.to_tensor([194], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_385f8fecc1f9618308f6eee5a79ba480(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [194]
        input_2 = [195]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_86c0bcfb69571c5c79ffaef40a03e2ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_385f8fecc1f9618308f6eee5a79ba480
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([194], dtype='int64').reshape([1]),
            paddle.to_tensor([195], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_2332a7a0678d82a36f928be698792eb5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_1 = [195]
        input_2 = [196]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_de27ae82a7f601d83e82d93f37760e2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2332a7a0678d82a36f928be698792eb5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([195], dtype='int64').reshape([1]),
            paddle.to_tensor([196], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_597253aa86611e756c0f8d660f478c6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 8, 160, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b91eebc6c1575dbb5354e42d561223c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 8, 160, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_55e0d2e1b42478295507283fee130bdd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6e81743cb1d0700dfd04ea418e38590
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 8, 160, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b57bab32b9e6bddc48fb931be97f46e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ee20b24d0dc5c0f8dfc8a85b251f263
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_78eb26e388a2440bcff880c3a0b71d0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab25d3d5c606d444873ae972ad49a88d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_df174cdf4c08ef88b0f556996fde95a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_206bb51737a40e49bcfd02c33c6f47a8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1b5e5577a3108804f61dbcb85744118a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60c3a53dc70160884ae36824e2bb7ab9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d1538daf6f038c240ebf117663bb59d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ea652a9b28ef535190e472ee073bfd6
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_20dd24d83262cdfddf0fbc8fe97a8d3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc217cebb3119779c263c0b26c1c0e92
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5917651b3e2c0a48026b638dd5b3b75b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65b631facec9038f7274627dd8b3d34b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_52a6e1a21068344a2972a9fc958a22eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1abc5a84c7b67e7998017479b702593
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3d688bd48af7a68811c269d5c561c555(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80551ea934c2d61c03fa6457d9ce1985
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_55ac8d1bfe8df81359b4f62550bc7ccf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7dfb4ada84d78f903e5538507d3b7ab
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b00bdcbc33d0c4ad6d5638bf5f6831f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91e429e4f0dfc5c2769ccfdc4607b6fa
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1b87b91a30ff5428dd7d27f0b2cfaaa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1262774c9edc7ff8f4ba3df5056d811
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_15b3ecff868a25b86970cfd519f06b1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b98c0cd022d0194b4c0832b0db60eac8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a9e82a5a457af4f16d6d6d2934c0ad85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aaac1a3979662b89e725d317a548ca91
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_13c488074ad8be2fef51d22ee06cd1f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d52063c93809e6ba0fe1e83fac1cc5b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bd53586e46c231715023eb146ec53035(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_33218763cd1a25505d53cd5b29035e0a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6cf4e5d49b5121f128154385bc099ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_786c42a7e462a07279f1e57ddde082e0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
            paddle.to_tensor([17], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1e2fd0dbe7a57d766e3f5d27ff7e76b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f768750d0804989fa348867c53b0bb1
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([17], dtype='int64').reshape([1]),
            paddle.to_tensor([18], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_972c5ed67d0562bdda629133023869da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aab8b254011b8ce61407c27f0f4c642f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([18], dtype='int64').reshape([1]),
            paddle.to_tensor([19], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a0ea204706a649c61e030c4bcf720ec3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adacbc4e820398794baec64fd98a3eb5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([19], dtype='int64').reshape([1]),
            paddle.to_tensor([20], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f3f9d9cdf78f2f467fd384dac72d5d47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c129e38203897bdcdc39d76ab610abe6
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([20], dtype='int64').reshape([1]),
            paddle.to_tensor([21], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_89b47a3a02d6411535431eb55f503b30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60c299b11353f17bec6c5760b2ec71e6
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([21], dtype='int64').reshape([1]),
            paddle.to_tensor([22], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d39ecdd23d6c7450b0fd6e42c953d1dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd1b9dfb33b0bf8fc2956226a617717c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([22], dtype='int64').reshape([1]),
            paddle.to_tensor([23], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4f037d0bebcc63dd787bdece6ae9447e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f598268b6c5d69f0bfc5945c4ef367f5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([23], dtype='int64').reshape([1]),
            paddle.to_tensor([24], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_45f50887ee6ae038b18fba7f4b6b9233(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fc7f50d72d3a27bf2d77c6cfb11617a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([24], dtype='int64').reshape([1]),
            paddle.to_tensor([25], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_df1e54bac4c0c21fa1f687d30567cfef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97dfa6384f0e00834b4731f732250d7c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([25], dtype='int64').reshape([1]),
            paddle.to_tensor([26], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1f53368c772adeb55c3b637ca156edda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86b557c5989b47fdac5f937cb60ef377
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([26], dtype='int64').reshape([1]),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_53588342e2c13c56733ff594037ce939(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf328c3e1784a95d1c8c5174bbfc5ba
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
            paddle.to_tensor([28], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c6b169e1e3cbadad3b9d8cea9bdae122(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b66c62fc3e483a566edf5e8f981ea3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([28], dtype='int64').reshape([1]),
            paddle.to_tensor([29], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_def1d252a6c27a42e7ea6aff69752fdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4ba03ff4391abf9fd557ef65dd6325e
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([29], dtype='int64').reshape([1]),
            paddle.to_tensor([30], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_35632dc91ae02f72022ed282417197b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9f54fa33b7dff41838d02ed1f33cc2
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([30], dtype='int64').reshape([1]),
            paddle.to_tensor([31], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cd972f581931410e18645228b31e0e6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d35009155b9636cfbd28ba3893f672b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([31], dtype='int64').reshape([1]),
            paddle.to_tensor([32], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2cf51874b56b2ff4bce139d58af05d59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5a686f212739ed486f6ca0e7a1c731d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([32], dtype='int64').reshape([1]),
            paddle.to_tensor([33], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f5369b16b33b361026cf4e74b0883b28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7edebd0037dd4b4d9b8bb90546382d1a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([33], dtype='int64').reshape([1]),
            paddle.to_tensor([34], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_afd3c0a12b9507783344064578de1910(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c769d7c7b73b95aa762264fb20471cae
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([34], dtype='int64').reshape([1]),
            paddle.to_tensor([35], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_07f02f39a77096314dbed28ea5007a47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1db492ab74727de620b8a48a35c176b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([35], dtype='int64').reshape([1]),
            paddle.to_tensor([36], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7bde95a94b75559305ea6d9c574f4f2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43d5b23888ac8f77827eaf5ea24e488c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([36], dtype='int64').reshape([1]),
            paddle.to_tensor([37], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e7e5ef935e97a73e05a6a25f9917a4ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf0e593fb3d8c20762680d907541d9ca
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([37], dtype='int64').reshape([1]),
            paddle.to_tensor([38], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e6d6fe5f7020748dc760fb0dcbdb301b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8f3e6a0312f63512f7e28c470bd5b42
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([38], dtype='int64').reshape([1]),
            paddle.to_tensor([39], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0db0f1cf18a674643cb19553fa14d721(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6cb44aeab8767351869c59ba5c722f9c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([39], dtype='int64').reshape([1]),
            paddle.to_tensor([40], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d8003734b27f53be7ab09a6f1e2643bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55e97202eb36c293cb8f7f714219eadc
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([40], dtype='int64').reshape([1]),
            paddle.to_tensor([41], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4560e6f5076c6c81f918daaff1ab8c71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2fe4027be3f35625d7febc50a58d4429
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([41], dtype='int64').reshape([1]),
            paddle.to_tensor([42], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5f25f23966fb2449fc4726ec1b1055c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa07965657440e2272b2339a117b9b51
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([42], dtype='int64').reshape([1]),
            paddle.to_tensor([43], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9e3df446767136cb0a9008b1542edc29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c25c4e732671e4d40e22bb7b9fd7c26
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([43], dtype='int64').reshape([1]),
            paddle.to_tensor([44], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2882428916feb4f6e1f631ae9630cd30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e94d2aba926c9fa13ac0e1831830c67
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([44], dtype='int64').reshape([1]),
            paddle.to_tensor([45], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a9b3809a31324255d2871227566a8e67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a31b0d0b621cb336425463df47f5a24
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([45], dtype='int64').reshape([1]),
            paddle.to_tensor([46], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_87f77f54c7d56bd6d94cb80e3179d9bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d37c28e6918cbf046fa5642b8cd12d8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([46], dtype='int64').reshape([1]),
            paddle.to_tensor([47], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f0d802d0bc83edeafede9457cc1dcec6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c334926e6aca226b4a3efe04f7c96842
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([47], dtype='int64').reshape([1]),
            paddle.to_tensor([48], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bdcaa293f5cd87f5f24cc626e6a90bd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_990862c82a422dc21fe38251ec184b68
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([48], dtype='int64').reshape([1]),
            paddle.to_tensor([49], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_31191d9faf48fc321b18de76715495da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ee20b24d0dc5c0f8dfc8a85b251f263
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3b365bee10e46b34dc71bc33cbbb33ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab25d3d5c606d444873ae972ad49a88d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_738a9b51d34608009dfba8c7520559a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_206bb51737a40e49bcfd02c33c6f47a8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3d4b843f61df4d3b82926a01c3e0f981(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60c3a53dc70160884ae36824e2bb7ab9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a43847032bba998734050dbe5dd15918(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ea652a9b28ef535190e472ee073bfd6
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4ed546ca68af11a35403a58094a8437b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc217cebb3119779c263c0b26c1c0e92
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_85755e379590ad4fd699873c3f06932b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65b631facec9038f7274627dd8b3d34b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8d28a71a605d0845bbd0fbeb966b7ed2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1abc5a84c7b67e7998017479b702593
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_20c0083a7d8fb93b7da91dc10e547c08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80551ea934c2d61c03fa6457d9ce1985
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d5ad41d9397944f573a49500d4ca2285(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7dfb4ada84d78f903e5538507d3b7ab
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f597baa4ba5e69af0a5bb0e409f301ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91e429e4f0dfc5c2769ccfdc4607b6fa
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_84a99ca604fddb0ea219ee0f0d7255b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1262774c9edc7ff8f4ba3df5056d811
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5e06b7198a22b13af7d6efc835e49502(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b98c0cd022d0194b4c0832b0db60eac8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_75656440d73d86c0ee305e9d90742131(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aaac1a3979662b89e725d317a548ca91
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b4a4429d83eea7a1244fa0faf63297d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d52063c93809e6ba0fe1e83fac1cc5b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_49928415e7c3b86347b7e40e5003d0c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_33218763cd1a25505d53cd5b29035e0a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_75d750c675068cc57c63ae6d9454b88c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1174, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f34035a3e9c24674db07ca656d3a7908(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1174, 384], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e5333e10da3af43581127a64fcebe342(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 6, 1174, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_89c7c2b6dd2778bf1737a6a685a9be29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 6, 1174, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f833e387f5e1a9494b5538e735b53596(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6e81743cb1d0700dfd04ea418e38590
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 6, 1174, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_03e8168d8363f28ed1dda2b9a722b2a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2da7ae3c8d29b3ed7515fe0ddc9e1505(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a75637a1fef41c1e368868c6e9fde5e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 256, 8, 16], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a2bcfb091a042f5897318132265a7053(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 256, 8, 16], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_62165a7d84f9a69fb699ae3bafac604a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 256, 8, 16], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a9f271b691ef42a670cce3ae7d2e37c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a051d3073974810fbb5d48c6cc0c7f6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6faaa108eae80dd038f9d942025ebf40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 116, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4d894d7eb39d55cffe71e4be4c6343be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 116, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6255760d7a3f9e82428ffe1b431d76c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 116, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_80c1c83eaa81eb2406d807d65b315989(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 116, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_42f0f5e3f4e1eb0a0834536340499f62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1174, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_55c158f96b8812b5ba29207df42ae523(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1174, 768], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cd7d735ed7026423b2c3fa145d2ca8b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 1174, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f02ca84996e7d5fa1a0ff8da281e6f1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 1174, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_987dd183fff2182ec072d3d6ededf8b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6e81743cb1d0700dfd04ea418e38590
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 1174, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_169e30eafe356b29b5d6982770d8a769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_711cb0502cbef1bd1ece6d36dda022c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e04773b902bf61052d6383fa93a40d1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 768, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a1ec56d12ee58241ed41978843b3c95e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 768, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_805ae1cf8f1e1ad18ef567476a2e4df7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 768, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_19989624090fd78511bfdbe171839aa8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 768, 32, 32], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f78c3e556db4d28d93b13dbc7e495665(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 384, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_76fad67301269b9dacf1bbfcf1dbd9da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 384, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1c4b07e53f5d7b64f0b65c3619139f92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 384, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_14c11d4fa684feea3db244f956c135aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 384, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2a55bce2fefb07fcfb41f7d07bec96f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8b55443b5023d6ed0ac6cd504aa6bc34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_670aff0e95dc0c0aeb4983c2dc3405b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c322cb82df24a501aab30f9f4b815a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a2c3355e8728034f7693511380847761(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb5284f45247c5f2fad7c0d6a26f7b5
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 128, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_53f2d292b8f704e76509406420c063ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2c4fe2e3b8be465edffea9f38bccf4ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe8eadc3d87caa62ce54669638983b5
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 3136, 96], dtype='int32').reshape([3]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a2ffa2b28e11ef87d2077af5a2de7252(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bad83a22783fb1303d0eea6272a1b4b
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 56, 56, 96], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_965148594f7c6cdffd11793ca7445cd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655a8853d524fc358a7a115c0cdb6310
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 8, 50, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7ee7a48fe6323d9dc1a9f4299152d808(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d755a5918f1fbe9c57ad683e82350321
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 8, 50, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f7c8232ce96e996c6920efb02de59fc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6e81743cb1d0700dfd04ea418e38590
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 8, 50, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()