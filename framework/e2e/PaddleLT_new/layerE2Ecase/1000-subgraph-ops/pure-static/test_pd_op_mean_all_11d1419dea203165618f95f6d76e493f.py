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



class PrimitiveOp_73e799f11a08ee0bcc474abac15e2d9b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d6ce7e8d20fc4d34b4cb888cd4132ee1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73e799f11a08ee0bcc474abac15e2d9b
    def get_inputs(self):
        return [
            paddle.uniform([4, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_62a5de42bf478e259aab45f639dca144(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3d067fbc39bc619889d50242ea8791f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62a5de42bf478e259aab45f639dca144
    def get_inputs(self):
        return [
            paddle.uniform([3, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c6105d23324fda249477d3d7dd73ae1c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1758, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8b202d7dea0c8657869d0a91ea9f9539(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6105d23324fda249477d3d7dd73ae1c
    def get_inputs(self):
        return [
            paddle.uniform([1758, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1612c70ffbf3de36d63d393057fa5878(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5593, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_080b733207730995f583365057e0346e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1612c70ffbf3de36d63d393057fa5878
    def get_inputs(self):
        return [
            paddle.uniform([5593, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_496f794238a4552e832b81560c2b2e1b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1763, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b91296bba776a10610099fdfdb04d260(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_496f794238a4552e832b81560c2b2e1b
    def get_inputs(self):
        return [
            paddle.uniform([1763, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fe7e927cdf586a5e0844b3a54bd79424(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1490, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7b8f2e5586c4605b737845787dc17c29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe7e927cdf586a5e0844b3a54bd79424
    def get_inputs(self):
        return [
            paddle.uniform([1490, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e6bdeecdd3d6da0cbe2986965b34ec3e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2076, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cba622330956bf0c05b016a06b88f1e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6bdeecdd3d6da0cbe2986965b34ec3e
    def get_inputs(self):
        return [
            paddle.uniform([2076, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_46051f150cf7c5cb27d1559d14e9ebc6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4642, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_11c43826e5158368ba213b1a16a9988b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46051f150cf7c5cb27d1559d14e9ebc6
    def get_inputs(self):
        return [
            paddle.uniform([4642, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f0f6b48512914285ca1fbf6b8f068334(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e8e0c66dbf000b9d69be376ce6190e19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0f6b48512914285ca1fbf6b8f068334
    def get_inputs(self):
        return [
            paddle.uniform([6, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_073e29b7fd533b3109c6cc18ca64ecdb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1047, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_591a0b7eae01566a758fb6a5aadc207f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_073e29b7fd533b3109c6cc18ca64ecdb
    def get_inputs(self):
        return [
            paddle.uniform([1047, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e57fce71fae112006e59e3e4420e44de(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2359, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cf07c6b086ed4e7a123f6d9e35a24236(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e57fce71fae112006e59e3e4420e44de
    def get_inputs(self):
        return [
            paddle.uniform([2359, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e7dfe87e650c4a7a557bca89c1630c30(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3049, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0380b39c343b226ccc2f9c995e2a9873(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7dfe87e650c4a7a557bca89c1630c30
    def get_inputs(self):
        return [
            paddle.uniform([3049, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4921f086c385a1a0832ba274930bff06(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3806, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_404d9df44add42687ffa3c4c97475593(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4921f086c385a1a0832ba274930bff06
    def get_inputs(self):
        return [
            paddle.uniform([3806, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_64e8b480d6c6ff97bb12729c51d14469(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3014b352d7b5f68dbb59af9086b1a87a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64e8b480d6c6ff97bb12729c51d14469
    def get_inputs(self):
        return [
            paddle.uniform([2, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4c93162b7e733e3e804e376fac850d63(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2054, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f0840e081df79d1b0383ee23af60d8ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c93162b7e733e3e804e376fac850d63
    def get_inputs(self):
        return [
            paddle.uniform([2054, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_78167757c4bc9871525aca32a65bfdb7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean_all(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4218, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6dc092b224b3dcf67eeb068b0e39aa14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78167757c4bc9871525aca32a65bfdb7
    def get_inputs(self):
        return [
            paddle.uniform([4218, 4], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()