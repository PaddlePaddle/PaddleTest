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



class PrimitiveOp_4ede20f3bf44f0b5cb83ff404f364d02(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 16, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1b3de4b5b8c1de592b3d311b5dd3f247(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ede20f3bf44f0b5cb83ff404f364d02
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 16, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_8093a796ccfa2e4c3ff6d30a3d612d27(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 40, 40], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 40, 40], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 40, 40], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f2c3503645a6bf79a4ece9464015311e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8093a796ccfa2e4c3ff6d30a3d612d27
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 40, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 40, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 40, 40], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_8269b85a8df50fee55647480becf4444(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [0, 0], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 120, 200], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 2, 120, 200], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1, 120, 200], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5bbb3383d505d9bd76d1f5fb3ceb4d63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8269b85a8df50fee55647480becf4444
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 120, 200], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 2, 120, 200], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1, 120, 200], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_66c5c3d1675a22bf7b2483c7961b744e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 258, 12, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 12, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 258, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 12, 12], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3f93154351d1a8ff1355abe97d585485(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66c5c3d1675a22bf7b2483c7961b744e
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 258, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 12, 12], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_eced543f688f4120391a9bb516e491c1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 12, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 12, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 12, 12], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ebb5db0a104b95a9a6c1e612e1fa042c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eced543f688f4120391a9bb516e491c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 12, 12], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_22f4a93279021570a95add733a321466(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 30, 50], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 30, 50], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 30, 50], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_76dcba64601bb17171be399f38b51c50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22f4a93279021570a95add733a321466
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 30, 50], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 30, 50], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 30, 50], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_77c3b6b42e7455d04f0561b0378d0ef0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 48, 72], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 48, 72], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 48, 72], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5e4ca2aa1b971869a44c0a0bc550418c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c3b6b42e7455d04f0561b0378d0ef0
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 48, 72], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_da59bd7cd164dc5918d525a1e47b4f7c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 96, 144], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 96, 144], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 256, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 96, 144], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9c75bd2cbc308a6adc6e541facc7c324(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da59bd7cd164dc5918d525a1e47b4f7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 96, 144], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d7d03c7be36000033d08f5f1f2db4a37(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 112, 160], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 112, 160], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 112, 160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ea5a1a03a3154206e9b9ea03f368bbd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7d03c7be36000033d08f5f1f2db4a37
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 112, 160], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 112, 160], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 112, 160], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e1c19f765a557e63ef8c3b0f9ac51290(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 258, 40, 40], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 40, 40], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 258, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 40, 40], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_19699df01e80c4498ed47ab7f790cf33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1c19f765a557e63ef8c3b0f9ac51290
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 40, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 40, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 258, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 40, 40], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c8a364cd42cb594bd6481f4cf93cc56a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 12, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 12, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 12, 12], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5574c6817313da67cdab1c1b7798761a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8a364cd42cb594bd6481f4cf93cc56a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 12, 12], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_4e4c44ea11b2fcaca1cf215ed992973b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 48, 72], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 48, 72], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 256, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 48, 72], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6d61115744637549c03387eca9986d95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e4c44ea11b2fcaca1cf215ed992973b
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 48, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 48, 72], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_403538f1247599a39ac164282d97a77e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 60, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 60, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 256, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 60, 100], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a030907a727ce57c2cb14f05f40dd24c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_403538f1247599a39ac164282d97a77e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 60, 100], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 60, 100], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 60, 100], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_38eb38d2fa64a09775dc2f04c879a432(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 258, 15, 25], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 15, 25], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 258, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 15, 25], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5ea57fefff5ec443bc58f752977cf8e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38eb38d2fa64a09775dc2f04c879a432
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 15, 25], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 15, 25], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 258, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 15, 25], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5fcd5d427d0659878e143b1bf99ee7ec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 24, 24], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 24, 24], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 24, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8e6967d05562f2a0f70917173710dc0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fcd5d427d0659878e143b1bf99ee7ec
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 24, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_cad9ca7829891306baaf87c5bfb29c78(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 56, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 56, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 56, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8989fad319ff224cc2a3960b701c0258(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cad9ca7829891306baaf87c5bfb29c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 56, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 56, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 56, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_eef25c1bea04d3f4ae40c0dac79792de(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 258, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 258, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 16, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_71ddc66e8a3aba9214ce3b4679a1054b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eef25c1bea04d3f4ae40c0dac79792de
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 258, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 16, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d46ba484762ff5228a49f40f35453386(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 192, 288], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 192, 288], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 256, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 192, 288], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7ccca136ba8c009a976371a75a6d211a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d46ba484762ff5228a49f40f35453386
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 192, 288], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 192, 288], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 192, 288], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ff9e7d3209741a05c00a85f5144bf849(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 36, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 36, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 36, 36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ea6eee244ca20c9c6225d9ed4d05defb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff9e7d3209741a05c00a85f5144bf849
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 36, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 36, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 36, 36], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9277136e1fe25b9542a9fa8f69dd5bbc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 7, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 7, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 7, 10], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_db3dfaee5f6045318b913a34da7a66b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9277136e1fe25b9542a9fa8f69dd5bbc
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 7, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 7, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 7, 10], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ea1d62e8cfff7b339cb3d72c2dffec47(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 14, 20], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 14, 20], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 14, 20], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8010b9ba579c6c5889b0bedc7208ea8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea1d62e8cfff7b339cb3d72c2dffec47
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 14, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 14, 20], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d3d24f6b9a5d43454436409e8bee91c5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 258, 36, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 36, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 258, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 36, 36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f5f4d8eb57137cda9da0e6d9a1650870(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3d24f6b9a5d43454436409e8bee91c5
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 36, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 36, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 258, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 36, 36], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_78adf9e645892faed7c02af7123b0cea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 60, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 60, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 60, 100], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_683e0bb22f07ffabfda576454caa805f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78adf9e645892faed7c02af7123b0cea
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 60, 100], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 60, 100], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 60, 100], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_065d81614285046fd8257e19b2fc5a37(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 30, 50], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 30, 50], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 256, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 30, 50], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f652e04fe58229899de6e14b2add8686(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_065d81614285046fd8257e19b2fc5a37
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 30, 50], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 30, 50], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 30, 50], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6d26103b93c53b08916570ed812c4e58(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 120, 200], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 120, 200], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 256, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 120, 200], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5bf96d27c6c18ace8a27c2cc30522693(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d26103b93c53b08916570ed812c4e58
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 120, 200], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 120, 200], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 120, 200], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_993321693368218515b350a8a3219f0e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 258, 24, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 24, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 258, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 24, 36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c8a54c56347b46045583849b8f957f25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_993321693368218515b350a8a3219f0e
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 258, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 24, 36], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e61b7a7b3615154d03d639a9574085ca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 40, 40], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 40, 40], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 40, 40], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6eefb498f550ec2cdf5d205469ed3b83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e61b7a7b3615154d03d639a9574085ca
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 40, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 40, 40], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c41eb1ca2956fb3ff921a07279ea1c1a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 28, 40], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 28, 40], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 28, 40], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_74dd90e855ea9ab2ab618194987f14e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c41eb1ca2956fb3ff921a07279ea1c1a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 28, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 28, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 28, 40], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a4a90827d74142ef63af242224a6d78e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 96, 144], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 96, 144], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 96, 144], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_11ed8b299287b224fb2e8bd54e02c509(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4a90827d74142ef63af242224a6d78e
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 96, 144], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 96, 144], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_806bf06961da2c1c7f7612eedd76747e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 258, 24, 24], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 24, 24], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 258, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 24, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e24497fdd5801b93ba0215bc4de8e186(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_806bf06961da2c1c7f7612eedd76747e
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 258, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 24, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_2d868038e364b5f87ca83e0bcdf00474(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 24, 24], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 24, 24], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 24, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c4673585a8eb9ff680b6640bea30c8a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d868038e364b5f87ca83e0bcdf00474
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 24, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3f96e5dc030c20653c8c236eb73afd3d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 16, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1cf52bae4ee11b39f19a66d476be9b83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f96e5dc030c20653c8c236eb73afd3d
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 16, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1ebbfb91c21c7f0c0bc3dfd74081afea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 36, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 18, 36, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9, 36, 36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8bf44167fc7045f1435bc39b511d03c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ebbfb91c21c7f0c0bc3dfd74081afea
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 36, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 18, 36, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9, 36, 36], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_148718eab405307e3cf3d8395539469f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [0, 0], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 192, 288], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 2, 192, 288], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 128, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1, 192, 288], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7b1f1a056ca59a9fb47c79eb7e775811(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_148718eab405307e3cf3d8395539469f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 192, 288], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 2, 192, 288], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1, 192, 288], dtype='float32', min=-0.5, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()