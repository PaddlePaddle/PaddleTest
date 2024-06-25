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



class PrimitiveOp_cf817579bb8a2762768aa16116d24195(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300, 256, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[300, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bfb9f72fc27040510599d0cfd0cb800f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf817579bb8a2762768aa16116d24195
    def get_inputs(self):
        return [
            paddle.uniform([300, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[300, 1], dtype='int32'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_6cc77524726a107415299ec54fb2a237(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8, 256, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[8, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3e7f2d54bdc6e52c3f4a25ba1b7ea3f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6cc77524726a107415299ec54fb2a237
    def get_inputs(self):
        return [
            paddle.uniform([8, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6], [7]], dtype='int32').reshape([8, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_cf18d268c4133c6aa5e2d310eaf3e7c9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 256, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[2, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f897340c0041b2f67c6c47c012b8fef5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf18d268c4133c6aa5e2d310eaf3e7c9
    def get_inputs(self):
        return [
            paddle.uniform([2, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [1]], dtype='int32').reshape([2, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_f839561a6f8b25fd3f0e332bd5d6f96f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 256, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[100, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_654ba34725d3562969bdf195a31f13c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f839561a6f8b25fd3f0e332bd5d6f96f
    def get_inputs(self):
        return [
            paddle.uniform([100, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[100, 1], dtype='int32'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_a7e4a5ff01a60042f2a0c6484e37ba05(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[2100], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6f26b762d5cb8ef09e54f1d2f9ebdf9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7e4a5ff01a60042f2a0c6484e37ba05
    def get_inputs(self):
        return [
            paddle.to_tensor([3], dtype='int32').reshape([1]),
            paddle.randint(low=0, high=3, shape=[2100], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_82a7fe4184affbaea1e9f7856b21dab9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[2100], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f9a3537821fafe2f54c1445cadbac982(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_82a7fe4184affbaea1e9f7856b21dab9
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.15033258497714996, 0.038082554936409, 0.3704834580421448, 0.28550711274147034]], dtype='float32').reshape([1, 4]),
            paddle.randint(low=0, high=3, shape=[2100], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_f897340c0041b2f67c6c47c012b8fef5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf18d268c4133c6aa5e2d310eaf3e7c9
    def get_inputs(self):
        return [
            paddle.uniform([2, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [1]], dtype='int32').reshape([2, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_3a149d0b9c152cf5488d73b6db41e298(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[185691], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d28e613efda35015ae7e36c0d28d6cdd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a149d0b9c152cf5488d73b6db41e298
    def get_inputs(self):
        return [
            paddle.uniform([185691], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_feda2bd5a8dfbd61f66c4ef9e9e51e61(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[185691], dtype='int32'),
            paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a53ee41d58ed450f4df9aa45420771e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feda2bd5a8dfbd61f66c4ef9e9e51e61
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[185691], dtype='int32'),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_d96b5eaa8d3c02d84c9ba688c5e9f18a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[185691, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[8, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b6ee46f894406727157526fca9ef5e5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d96b5eaa8d3c02d84c9ba688c5e9f18a
    def get_inputs(self):
        return [
            paddle.uniform([185691, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1], [1], [5], [0], [9], [2], [4], [2]], dtype='int64').reshape([8, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b6ee46f894406727157526fca9ef5e5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d96b5eaa8d3c02d84c9ba688c5e9f18a
    def get_inputs(self):
        return [
            paddle.uniform([185691, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1], [1], [5], [0], [9], [2], [4], [2]], dtype='int64').reshape([8, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_f897340c0041b2f67c6c47c012b8fef5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf18d268c4133c6aa5e2d310eaf3e7c9
    def get_inputs(self):
        return [
            paddle.uniform([2, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [1]], dtype='int32').reshape([2, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_3e7f2d54bdc6e52c3f4a25ba1b7ea3f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6cc77524726a107415299ec54fb2a237
    def get_inputs(self):
        return [
            paddle.uniform([8, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6], [7]], dtype='int32').reshape([8, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_d4d14f2da2f8f848be987540fdf91294(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2], dtype='int32'),
            paddle.static.InputSpec(shape=[2002], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_030e750ecd2c2fb3afa493048e0bcdd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d14f2da2f8f848be987540fdf91294
    def get_inputs(self):
        return [
            paddle.to_tensor([9, 5], dtype='int32').reshape([2]),
            paddle.randint(low=0, high=3, shape=[2002], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_133238ab6cd6751fd5248f320291db40(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[21], dtype='int32'),
            paddle.static.InputSpec(shape=[1021], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3093cd514269b727bc858a921476152d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_133238ab6cd6751fd5248f320291db40
    def get_inputs(self):
        return [
            paddle.to_tensor([6, 0, 2, 8, 9, 6, 2, 5, 4, 0, 2, 4, 2, 2, 3, 5, 2, 4, 4, 1, 0], dtype='int32').reshape([21]),
            paddle.randint(low=0, high=3, shape=[1021], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_fd7838871003037d3a2652293ba16fe5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[242991], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c8446b119dd4d8b83cc7eff2e53f69d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd7838871003037d3a2652293ba16fe5
    def get_inputs(self):
        return [
            paddle.uniform([242991], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_6a07611383185a0f98249d29d2a528ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[242991], dtype='int32'),
            paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_70780e677ba27ee0af3acc1d082d2983(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a07611383185a0f98249d29d2a528ee
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[242991], dtype='int32'),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_59004accc7169fae6609253f30f8e0b8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[242991, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[5, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7dc66246dfb60fc481b65ba7a5b85fbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59004accc7169fae6609253f30f8e0b8
    def get_inputs(self):
        return [
            paddle.uniform([242991, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[8], [0], [6], [1], [5]], dtype='int64').reshape([5, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_7dc66246dfb60fc481b65ba7a5b85fbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59004accc7169fae6609253f30f8e0b8
    def get_inputs(self):
        return [
            paddle.uniform([242991, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[8], [0], [6], [1], [5]], dtype='int64').reshape([5, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_428bc119897bf1042f36167e55d844e3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[7, 256, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[7, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5031447b0c4f06c9dd523de03b183ad2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_428bc119897bf1042f36167e55d844e3
    def get_inputs(self):
        return [
            paddle.uniform([7, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6]], dtype='int32').reshape([7, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_4cb27e8bc5c6185d340d34347652e6db(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2], dtype='int32'),
            paddle.static.InputSpec(shape=[1002], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0b2550689a78f4b5b87765c125f6452b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4cb27e8bc5c6185d340d34347652e6db
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 5], dtype='int32').reshape([2]),
            paddle.randint(low=0, high=3, shape=[1002], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_c8c7a9db37fe8faf30f8a7c17a21d2b5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171888], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d56d6e3a395409774774587e45504e0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8c7a9db37fe8faf30f8a7c17a21d2b5
    def get_inputs(self):
        return [
            paddle.uniform([171888], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_7768db3c7a9ed67441af62ec9dec2f36(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171888], dtype='int32'),
            paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f04f721034ec79465961bd32d31d45f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7768db3c7a9ed67441af62ec9dec2f36
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[171888], dtype='int32'),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_bc010c512231ced6d2117b48f86515df(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171888, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[5, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_85ff9b571d4fe54413c9f3d9f6588ed3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc010c512231ced6d2117b48f86515df
    def get_inputs(self):
        return [
            paddle.uniform([171888, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[6], [4], [1], [4], [1]], dtype='int64').reshape([5, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_85ff9b571d4fe54413c9f3d9f6588ed3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc010c512231ced6d2117b48f86515df
    def get_inputs(self):
        return [
            paddle.uniform([171888, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[6], [4], [1], [4], [1]], dtype='int64').reshape([5, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_d90d05b9b7b897ba0ed9a5a22ef12f43(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 256, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[6, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2b3af9d6b41ca177926c9e917559c142(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d90d05b9b7b897ba0ed9a5a22ef12f43
    def get_inputs(self):
        return [
            paddle.uniform([6, 256, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [1], [2], [3], [4], [5]], dtype='int32').reshape([6, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d56d6e3a395409774774587e45504e0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8c7a9db37fe8faf30f8a7c17a21d2b5
    def get_inputs(self):
        return [
            paddle.uniform([171888], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_f04f721034ec79465961bd32d31d45f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7768db3c7a9ed67441af62ec9dec2f36
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[171888], dtype='int32'),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_865ab527dee2570291377802fdb0b915(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171888, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[7, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d85abdabbe699730926078b98ad907f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_865ab527dee2570291377802fdb0b915
    def get_inputs(self):
        return [
            paddle.uniform([171888, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[6], [4], [1], [4], [1], [3], [3]], dtype='int64').reshape([7, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d85abdabbe699730926078b98ad907f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_865ab527dee2570291377802fdb0b915
    def get_inputs(self):
        return [
            paddle.uniform([171888, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[6], [4], [1], [4], [1], [3], [3]], dtype='int64').reshape([7, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_c86fa3a9625735f4da585be7362e0aa5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 256, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[3, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c980464d86f1412739c11d3ce00cfd89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c86fa3a9625735f4da585be7362e0aa5
    def get_inputs(self):
        return [
            paddle.uniform([3, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [1], [2]], dtype='int32').reshape([3, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_3bdeb7ac00a4edbd59bf69e63655e77f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[217413], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_30d23445863943bbaf5651dbf23b9ca3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bdeb7ac00a4edbd59bf69e63655e77f
    def get_inputs(self):
        return [
            paddle.uniform([217413], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_36019249ee2d8875f89f7556768c1580(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[217413], dtype='int32'),
            paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d35a47f79bb49c8451becf3b37239f76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36019249ee2d8875f89f7556768c1580
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[217413], dtype='int32'),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_c205402b823905760facb533d36a46c5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[217413, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[103, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_01d2dd4457d82441f77ef8a3b419b9b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c205402b823905760facb533d36a46c5
    def get_inputs(self):
        return [
            paddle.uniform([217413, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[103, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_01d2dd4457d82441f77ef8a3b419b9b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c205402b823905760facb533d36a46c5
    def get_inputs(self):
        return [
            paddle.uniform([217413, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[103, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_f897340c0041b2f67c6c47c012b8fef5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf18d268c4133c6aa5e2d310eaf3e7c9
    def get_inputs(self):
        return [
            paddle.uniform([2, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [1]], dtype='int32').reshape([2, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_1d322e1b7820e56039c73adb356b6811(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_12ba1e3ce39c171d4390de7b74a3e2c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d322e1b7820e56039c73adb356b6811
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0]], dtype='int32').reshape([1, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_487020cd1656f85aa7353097272ee4db(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_057b912589e1d2f0a357c7ca133153cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_487020cd1656f85aa7353097272ee4db
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_0a501820d01bebcfd60c2e501e4f1aa6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2], dtype='int32'),
            paddle.static.InputSpec(shape=[3549], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_973a4e4b0e67ac2d4d7fac13ac108fa5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a501820d01bebcfd60c2e501e4f1aa6
    def get_inputs(self):
        return [
            paddle.to_tensor([6, 6], dtype='int32').reshape([2]),
            paddle.randint(low=0, high=3, shape=[3549], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_e56aa4115e8209b33823b841d5f1ab95(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[3549], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_584feabfcc36b243284b1521a2edb569(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e56aa4115e8209b33823b841d5f1ab95
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4723230302333832, 0.33346495032310486, 0.03808297589421272, 0.18030746281147003], [0.34793317317962646, 0.26936545968055725, 0.39894992113113403, 0.15911562740802765]], dtype='float32').reshape([2, 4]),
            paddle.randint(low=0, high=3, shape=[3549], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_35b304ad4804b31d6361efa8ff43fd45(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[7, 64, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[7, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cfeb0a19012ef478458a92a26a0295f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35b304ad4804b31d6361efa8ff43fd45
    def get_inputs(self):
        return [
            paddle.uniform([7, 64, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6]], dtype='int32').reshape([7, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_12ba1e3ce39c171d4390de7b74a3e2c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d322e1b7820e56039c73adb356b6811
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0]], dtype='int32').reshape([1, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_7a76543e721aa323b9b5da06791d1fc5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86970], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7550c28b5eb6907fefa9b52128044c97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a76543e721aa323b9b5da06791d1fc5
    def get_inputs(self):
        return [
            paddle.uniform([86970], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_5c242672c8dac286898772f4a80ce5db(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86970], dtype='int32'),
            paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7e68d73f0d10596e9141d7e21855d0de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c242672c8dac286898772f4a80ce5db
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[86970], dtype='int32'),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_5f6ce267310928ccaf153643689335e3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86970, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[6, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1c908c2152f6dc9575caea285adf800d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f6ce267310928ccaf153643689335e3
    def get_inputs(self):
        return [
            paddle.uniform([86970, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[9], [5], [1], [0], [0], [1]], dtype='int64').reshape([6, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_1c908c2152f6dc9575caea285adf800d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f6ce267310928ccaf153643689335e3
    def get_inputs(self):
        return [
            paddle.uniform([86970, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[9], [5], [1], [0], [0], [1]], dtype='int64').reshape([6, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_fc5704dcfec04864f089d79f251ddbaf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[205923], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3f7f36c446650d18bfd86a1b02050e0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc5704dcfec04864f089d79f251ddbaf
    def get_inputs(self):
        return [
            paddle.uniform([205923], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_34e3872cff83db406fe84f0713f49570(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[205923], dtype='int32'),
            paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_648d5d945a15c5420351bdf31c8ca5e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34e3872cff83db406fe84f0713f49570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[205923], dtype='int32'),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_6ffd093b03d6ac0a4776e5422dd95c78(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[205923, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[5, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_997cb0cebcfac576c9f96b0669699e6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ffd093b03d6ac0a4776e5422dd95c78
    def get_inputs(self):
        return [
            paddle.uniform([205923, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1], [0], [8], [4], [1]], dtype='int64').reshape([5, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_997cb0cebcfac576c9f96b0669699e6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ffd093b03d6ac0a4776e5422dd95c78
    def get_inputs(self):
        return [
            paddle.uniform([205923, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1], [0], [8], [4], [1]], dtype='int64').reshape([5, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_50537efa0ec605c85448f6ff777040af(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[153450], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a0a123588714962c43e7772173da62cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50537efa0ec605c85448f6ff777040af
    def get_inputs(self):
        return [
            paddle.uniform([153450], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_0193bc2da2aface07d1ca8ffed2e3a2b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[153450], dtype='int32'),
            paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_43df358a94b07f1a9b2e14d17cf37827(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0193bc2da2aface07d1ca8ffed2e3a2b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[153450], dtype='int32'),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_779eb8ed57b665baf8867f619451905c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[153450, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9faf8d89d76a107a97df4447d32d0546(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779eb8ed57b665baf8867f619451905c
    def get_inputs(self):
        return [
            paddle.uniform([153450, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[8], [4], [4], [2], [3], [1], [7], [4], [8], [3]], dtype='int64').reshape([10, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_9faf8d89d76a107a97df4447d32d0546(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779eb8ed57b665baf8867f619451905c
    def get_inputs(self):
        return [
            paddle.uniform([153450, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[8], [4], [4], [2], [3], [1], [7], [4], [8], [3]], dtype='int64').reshape([10, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_00744786f824e3e4ef838ce4828418ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5, 256, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[5, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_287802d93f1cdd6ca6cfb041b6543f35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00744786f824e3e4ef838ce4828418ee
    def get_inputs(self):
        return [
            paddle.uniform([5, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [1], [2], [3], [4]], dtype='int32').reshape([5, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_a6416dab56d3f81a98c525af6c81f5f6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[4116], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8a8303857a8e0fcb964c3bc1800d3a1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6416dab56d3f81a98c525af6c81f5f6
    def get_inputs(self):
        return [
            paddle.to_tensor([3], dtype='int32').reshape([1]),
            paddle.randint(low=0, high=3, shape=[4116], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_1c43c06fc82b3e99743b832495311eff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[4116], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fa16cf27b6b77142dc89f2c9e168f25d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c43c06fc82b3e99743b832495311eff
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.09953709691762924, 0.017059890553355217, 0.06178886815905571, 0.20930905640125275]], dtype='float32').reshape([1, 4]),
            paddle.randint(low=0, high=3, shape=[4116], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_5031447b0c4f06c9dd523de03b183ad2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_428bc119897bf1042f36167e55d844e3
    def get_inputs(self):
        return [
            paddle.uniform([7, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6]], dtype='int32').reshape([7, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_e7141b984b464473ee02e2693caa357f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[113061], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_713a750b6a3af2d57692e0de14fe953a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7141b984b464473ee02e2693caa357f
    def get_inputs(self):
        return [
            paddle.uniform([113061], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_7f96c81f223f74c8a0c76cd081f620d0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[113061], dtype='int32'),
            paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_42aa9c5fc4bace7d70826d2700308666(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f96c81f223f74c8a0c76cd081f620d0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[113061], dtype='int32'),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_c2a8e022f6cb750a9a61c0a92c0b5eca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[113061, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[4, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_549eed24f6ac559097a4622e8a62cac8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2a8e022f6cb750a9a61c0a92c0b5eca
    def get_inputs(self):
        return [
            paddle.uniform([113061, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[2], [6], [7], [8]], dtype='int64').reshape([4, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_549eed24f6ac559097a4622e8a62cac8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2a8e022f6cb750a9a61c0a92c0b5eca
    def get_inputs(self):
        return [
            paddle.uniform([113061, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[2], [6], [7], [8]], dtype='int64').reshape([4, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_5031447b0c4f06c9dd523de03b183ad2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_428bc119897bf1042f36167e55d844e3
    def get_inputs(self):
        return [
            paddle.uniform([7, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6]], dtype='int32').reshape([7, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_12ba1e3ce39c171d4390de7b74a3e2c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d322e1b7820e56039c73adb356b6811
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0]], dtype='int32').reshape([1, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_cd917d05b14ba00339d5da1e32c02208(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[123783], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_325fa687f9f7f4cd3829249ce48efd26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd917d05b14ba00339d5da1e32c02208
    def get_inputs(self):
        return [
            paddle.uniform([123783], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_71b1c26c5e83573a5f2db497c5c23fb4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[123783], dtype='int32'),
            paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a9c9f30cc4c7947dd0cf9f95b75e481f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71b1c26c5e83573a5f2db497c5c23fb4
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[123783], dtype='int32'),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_fcc3afcf0c994a5eb398628dd86e32be(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[123783, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[84, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d2cfb60de4426796e1a3616ad56fa911(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fcc3afcf0c994a5eb398628dd86e32be
    def get_inputs(self):
        return [
            paddle.uniform([123783, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[84, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d2cfb60de4426796e1a3616ad56fa911(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fcc3afcf0c994a5eb398628dd86e32be
    def get_inputs(self):
        return [
            paddle.uniform([123783, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[84, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_bfb9f72fc27040510599d0cfd0cb800f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf817579bb8a2762768aa16116d24195
    def get_inputs(self):
        return [
            paddle.uniform([300, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[300, 1], dtype='int32'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d28e613efda35015ae7e36c0d28d6cdd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a149d0b9c152cf5488d73b6db41e298
    def get_inputs(self):
        return [
            paddle.uniform([185691], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_a53ee41d58ed450f4df9aa45420771e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feda2bd5a8dfbd61f66c4ef9e9e51e61
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[185691], dtype='int32'),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_17e0a1864ecdc784c1f6711be8742338(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[185691, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[7, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7ee40dc7d081ef65458ae5276e0ae35b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17e0a1864ecdc784c1f6711be8742338
    def get_inputs(self):
        return [
            paddle.uniform([185691, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1], [1], [5], [0], [9], [2], [4]], dtype='int64').reshape([7, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_7ee40dc7d081ef65458ae5276e0ae35b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17e0a1864ecdc784c1f6711be8742338
    def get_inputs(self):
        return [
            paddle.uniform([185691, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1], [1], [5], [0], [9], [2], [4]], dtype='int64').reshape([7, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_287802d93f1cdd6ca6cfb041b6543f35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00744786f824e3e4ef838ce4828418ee
    def get_inputs(self):
        return [
            paddle.uniform([5, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [1], [2], [3], [4]], dtype='int32').reshape([5, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_a0a123588714962c43e7772173da62cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50537efa0ec605c85448f6ff777040af
    def get_inputs(self):
        return [
            paddle.uniform([153450], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_43df358a94b07f1a9b2e14d17cf37827(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0193bc2da2aface07d1ca8ffed2e3a2b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[153450], dtype='int32'),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_daafb21f55aaa6d84ae474185b62fc04(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[153450, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[6, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_95207ec3cc8ab1b538a65232bb4b739d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_daafb21f55aaa6d84ae474185b62fc04
    def get_inputs(self):
        return [
            paddle.uniform([153450, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[8], [4], [4], [2], [3], [1]], dtype='int64').reshape([6, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_95207ec3cc8ab1b538a65232bb4b739d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_daafb21f55aaa6d84ae474185b62fc04
    def get_inputs(self):
        return [
            paddle.uniform([153450, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[8], [4], [4], [2], [3], [1]], dtype='int64').reshape([6, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_f023d7187a00b0f25fe22fa10547efd6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_75e1258eb0d269518ce7a0f024674531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f023d7187a00b0f25fe22fa10547efd6
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_75e1258eb0d269518ce7a0f024674531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f023d7187a00b0f25fe22fa10547efd6
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_75e1258eb0d269518ce7a0f024674531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f023d7187a00b0f25fe22fa10547efd6
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_75e1258eb0d269518ce7a0f024674531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f023d7187a00b0f25fe22fa10547efd6
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_75e1258eb0d269518ce7a0f024674531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f023d7187a00b0f25fe22fa10547efd6
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_75e1258eb0d269518ce7a0f024674531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f023d7187a00b0f25fe22fa10547efd6
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_75e1258eb0d269518ce7a0f024674531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f023d7187a00b0f25fe22fa10547efd6
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_75e1258eb0d269518ce7a0f024674531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f023d7187a00b0f25fe22fa10547efd6
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_75e1258eb0d269518ce7a0f024674531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f023d7187a00b0f25fe22fa10547efd6
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_75e1258eb0d269518ce7a0f024674531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f023d7187a00b0f25fe22fa10547efd6
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_75e1258eb0d269518ce7a0f024674531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f023d7187a00b0f25fe22fa10547efd6
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_75e1258eb0d269518ce7a0f024674531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f023d7187a00b0f25fe22fa10547efd6
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_75e1258eb0d269518ce7a0f024674531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f023d7187a00b0f25fe22fa10547efd6
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_75e1258eb0d269518ce7a0f024674531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f023d7187a00b0f25fe22fa10547efd6
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_75e1258eb0d269518ce7a0f024674531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f023d7187a00b0f25fe22fa10547efd6
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_75e1258eb0d269518ce7a0f024674531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f023d7187a00b0f25fe22fa10547efd6
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_3ddebde075d0ebe56fdc5ec2fc779d72(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_235d29db0397938a2d5d308be198aab3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ddebde075d0ebe56fdc5ec2fc779d72
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 3], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_235d29db0397938a2d5d308be198aab3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ddebde075d0ebe56fdc5ec2fc779d72
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 3], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_2bdaa0438952e3bb4b39df463f0246be(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_79bd9c186ed893f9f7cb4d2d0208b107(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bdaa0438952e3bb4b39df463f0246be
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 9], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_79bd9c186ed893f9f7cb4d2d0208b107(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bdaa0438952e3bb4b39df463f0246be
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 9], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_b00a9ed12888346f4bd44e55d6065158(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4ab4aa0b1d21ad2e336a4866b2a858dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00a9ed12888346f4bd44e55d6065158
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0]], dtype='int32').reshape([1, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_12ba1e3ce39c171d4390de7b74a3e2c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d322e1b7820e56039c73adb356b6811
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0]], dtype='int32').reshape([1, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_80b4b2cb8a58b86dc2f73cb50e7516c7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[27], dtype='int32'),
            paddle.static.InputSpec(shape=[1027], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2e57214b8d4621ab364bbab296471ae4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80b4b2cb8a58b86dc2f73cb50e7516c7
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 2, 3, 5, 2, 4, 4, 1, 0, 6, 8, 6, 0, 6, 9, 3, 4, 9, 4, 0, 0, 7, 8, 6, 1, 9, 3], dtype='int32').reshape([27]),
            paddle.randint(low=0, high=3, shape=[1027], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_f40ad25b70b7bcb82497d69f34d44383(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8, 256, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[8, 1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_67403765336fa1923b227758cc598a23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f40ad25b70b7bcb82497d69f34d44383
    def get_inputs(self):
        return [
            paddle.uniform([8, 256, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6], [7]], dtype='int32').reshape([8, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_654ba34725d3562969bdf195a31f13c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f839561a6f8b25fd3f0e332bd5d6f96f
    def get_inputs(self):
        return [
            paddle.uniform([100, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[100, 1], dtype='int32'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_12ba1e3ce39c171d4390de7b74a3e2c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d322e1b7820e56039c73adb356b6811
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0]], dtype='int32').reshape([1, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_936b56eed857ebffe1d277d0b5724b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46216cc5d5134b7f772442a0a5c9ce3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_5878a5389b631111851ba93631e76a96(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[220968], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f47f8342e0778ef0038c1e30070ef8b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5878a5389b631111851ba93631e76a96
    def get_inputs(self):
        return [
            paddle.uniform([220968], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_465f3ab4d4d881acf401c81c89c92138(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[220968], dtype='int32'),
            paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b315461983fab0d42b7647ab0f382228(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_465f3ab4d4d881acf401c81c89c92138
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[220968], dtype='int32'),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_fe5352263da6889c3c4a23e2a065de94(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[220968, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[5, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_658c9296326ef204ab945807575f0171(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe5352263da6889c3c4a23e2a065de94
    def get_inputs(self):
        return [
            paddle.uniform([220968, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[6], [5], [2], [2], [8]], dtype='int64').reshape([5, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_658c9296326ef204ab945807575f0171(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe5352263da6889c3c4a23e2a065de94
    def get_inputs(self):
        return [
            paddle.uniform([220968, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[6], [5], [2], [2], [8]], dtype='int64').reshape([5, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_edc422e3538b0c872610add76650571e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b1f45c771c3a3a316d41b96fd5f74657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edc422e3538b0c872610add76650571e
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_46b1209e352bef23143834fbaef57181(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[16], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_59452af122ea8c7f7afe034b30eda05d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b1209e352bef23143834fbaef57181
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 2, 1, 2, 1, 0, 1, 2, 2, 2, 0, 2, 0, 0, 1, 1], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_fed3353868a25771d03fa09bfb00bf2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b1209e352bef23143834fbaef57181
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 2, 2, 1, 1, 2, 1, 0, 0, 2, 2, 1, 2, 1, 0, 1], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_f62e46d4f1f179abd76488adb0dcdec8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b1209e352bef23143834fbaef57181
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1, 0, 0, 1, 1, 2, 0, 2, 1, 2, 0, 1, 0, 2, 0], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b3c64b56e5e9d086747e8136b00494f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b1209e352bef23143834fbaef57181
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 0, 2, 0, 1, 1, 1, 0, 2, 2, 2, 0, 2, 2, 1, 2], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_9de6c80e99ccca3e945fe2486a3d9366(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b1209e352bef23143834fbaef57181
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 2, 1, 1, 1, 0, 0, 2, 1, 0, 0, 2, 2, 0, 1, 1], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_026cb13953fbeb9484489754c471ddeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b1209e352bef23143834fbaef57181
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1, 2, 0, 2, 1, 1, 1, 1, 0, 2, 2, 1, 1, 0, 1], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d1429d08a728704d075c3449b65ac01a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b1209e352bef23143834fbaef57181
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1, 1, 2, 2, 0, 2, 1, 0, 2, 2, 2, 2, 2, 1, 2], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_9938a2408869cd74020bf7d60bfe476d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b1209e352bef23143834fbaef57181
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 2, 0, 2, 2, 2, 0, 1, 2, 1, 1, 2, 0, 0, 0, 0], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_e0a14c47172c338e1cd51a225f736f33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b1209e352bef23143834fbaef57181
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 2, 0, 2, 0, 2, 1, 1, 0, 0, 0, 1, 0, 1, 1, 2], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_5dbba1eb5bb20405e7edfbcbf353ad60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b1209e352bef23143834fbaef57181
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 1, 1, 2, 2, 0, 0, 1, 0, 0, 1, 1, 2, 1, 1, 0], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_80c2e11ba769dc6740448616b6271065(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b1209e352bef23143834fbaef57181
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 2, 0, 1, 1, 1, 2, 2, 0, 0, 2, 2, 1, 1, 2, 0], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_2a687b9479ccad5df4c052031457e379(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b1209e352bef23143834fbaef57181
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 1, 1, 0, 1, 2, 2, 0, 2, 1, 0, 0, 2, 0, 2, 1], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8569388362c7a81a0353a67ed746650e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b1209e352bef23143834fbaef57181
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 0, 0, 2, 1, 1, 0, 1, 2, 2, 1, 2, 2, 1, 1, 1], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_f69b1e2b61bf9bc77e3eb9fb943fdca4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b1209e352bef23143834fbaef57181
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1, 0, 1, 0, 2, 1, 1, 0, 0, 0, 2, 2, 1, 1, 1], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_30ec6cf5a832d85a9609c4f5c37c4535(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b1209e352bef23143834fbaef57181
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 0, 0, 1, 0, 2, 2, 2, 2, 0, 2, 0, 1, 0, 2, 1], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_2abe540975cc285c0389fb0135f984b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b1209e352bef23143834fbaef57181
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 0, 1, 1, 2, 1, 0, 2, 2, 2, 1, 0, 1, 1, 2, 1], dtype='int64').reshape([16]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_c2ca022b71eae0edc4af1e0a77d6630a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[185658], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3c43756161e3389f0c99d5f137700c7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2ca022b71eae0edc4af1e0a77d6630a
    def get_inputs(self):
        return [
            paddle.uniform([185658], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_3365643ad37b0e0c70e9f3b36fe9e6d0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[185658], dtype='int32'),
            paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0811490e1f13cb682f5a72c36aadc6f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3365643ad37b0e0c70e9f3b36fe9e6d0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[185658], dtype='int32'),
            paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_6675574bb21c1a6d4e38a768300e1f49(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[185658, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[7, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cc648255662829597d5b028c86020dfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6675574bb21c1a6d4e38a768300e1f49
    def get_inputs(self):
        return [
            paddle.uniform([185658, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[9], [1], [6], [9], [2], [8], [2]], dtype='int64').reshape([7, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_cc648255662829597d5b028c86020dfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6675574bb21c1a6d4e38a768300e1f49
    def get_inputs(self):
        return [
            paddle.uniform([185658, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[9], [1], [6], [9], [2], [8], [2]], dtype='int64').reshape([7, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_5031447b0c4f06c9dd523de03b183ad2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_428bc119897bf1042f36167e55d844e3
    def get_inputs(self):
        return [
            paddle.uniform([7, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6]], dtype='int32').reshape([7, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_2b3af9d6b41ca177926c9e917559c142(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d90d05b9b7b897ba0ed9a5a22ef12f43
    def get_inputs(self):
        return [
            paddle.uniform([6, 256, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [1], [2], [3], [4], [5]], dtype='int32').reshape([6, 1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()