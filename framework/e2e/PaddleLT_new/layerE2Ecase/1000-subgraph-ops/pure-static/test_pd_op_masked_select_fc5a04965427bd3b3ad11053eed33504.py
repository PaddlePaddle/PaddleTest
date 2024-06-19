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



class PrimitiveOp_b840ef8f850d593870bcb4a88709539e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.masked_select(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3024, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3024, 4], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c88f22670082ddb721985a41ec51bb35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b840ef8f850d593870bcb4a88709539e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_c88f22670082ddb721985a41ec51bb35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b840ef8f850d593870bcb4a88709539e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024, 4], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_bae8311eec31df5baf25ab58b963ac0b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.masked_select(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3024], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3024], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b03e0a5bf0750139a0348a864cde8cd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bae8311eec31df5baf25ab58b963ac0b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_b8e1b7638e9d39ac87caaca07cd58dba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.masked_select(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3024, 68], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3024, 68], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2310fada9f3c40652c336b6330f6d5e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8e1b7638e9d39ac87caaca07cd58dba
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024, 68], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_c88f22670082ddb721985a41ec51bb35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b840ef8f850d593870bcb4a88709539e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024, 4], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_12dd2f9299ce59363be4f464fc20d9f5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.masked_select(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4725, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4725, 4], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c8acc4738f86fb3342c870f17b7ba2fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12dd2f9299ce59363be4f464fc20d9f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_c8acc4738f86fb3342c870f17b7ba2fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12dd2f9299ce59363be4f464fc20d9f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725, 4], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_a9ef836e016d5e9d0c35fa8ee9bfc97c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.masked_select(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4725], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4725], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7c095afa17e24b5acab786973d2ec319(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9ef836e016d5e9d0c35fa8ee9bfc97c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_b99a1c52c79587db32c4ca537d05011a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.masked_select(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4725, 68], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4725, 68], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ef8babaf949f8f91d29c78bfe87acd5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b99a1c52c79587db32c4ca537d05011a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725, 68], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_c8acc4738f86fb3342c870f17b7ba2fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12dd2f9299ce59363be4f464fc20d9f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725, 4], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_cbb4f55945237db5c3a152223b761db2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.masked_select(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 500, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 500, 128], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8ee28deeef36cef8056dd674f8ea518b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbb4f55945237db5c3a152223b761db2
    def get_inputs(self):
        return [
            paddle.uniform([1, 500, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 500, 128], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_8ee28deeef36cef8056dd674f8ea518b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbb4f55945237db5c3a152223b761db2
    def get_inputs(self):
        return [
            paddle.uniform([1, 500, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 500, 128], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_3f62652d33ddee7084a1803608f665e1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.masked_select(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4116, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4116, 4], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_27d9574c0f15c943aca65f8fe5260423(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f62652d33ddee7084a1803608f665e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_27d9574c0f15c943aca65f8fe5260423(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f62652d33ddee7084a1803608f665e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_6c9b8d57a736fcff29da64d47a46c259(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.masked_select(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4116], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4116], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fad27e96c460bdf4249fbd82e2ad02f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c9b8d57a736fcff29da64d47a46c259
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_034cae32af7842485cabceeecb8133ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.masked_select(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4116, 68], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4116, 68], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6f5d8b7a09ee8d823278a0aef0ab1cde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_034cae32af7842485cabceeecb8133ee
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 68], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_27d9574c0f15c943aca65f8fe5260423(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f62652d33ddee7084a1803608f665e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_382bcf1a3ffc3879e89d6852a3cf27d5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.masked_select(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3549, 4], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_96cc95f8242fe2658aa2d750b361a0d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_382bcf1a3ffc3879e89d6852a3cf27d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_96cc95f8242fe2658aa2d750b361a0d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_382bcf1a3ffc3879e89d6852a3cf27d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_3c336066d6758a64b16d49d83984ac67(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.masked_select(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3549], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d9c4c468f0a3f91d4bdcc30898c699ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c336066d6758a64b16d49d83984ac67
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_3377d32850886d47327d06865727f766(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.masked_select(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549, 76], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3549, 76], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_03c06726ddb48f639cd4ad93ae0e7e41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3377d32850886d47327d06865727f766
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 76], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_96cc95f8242fe2658aa2d750b361a0d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_382bcf1a3ffc3879e89d6852a3cf27d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_7cb7eb8034023a23ab1e0131d9a618c1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.masked_select(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2434, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 2434, 4], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_755c1dac97994751b74dc06b1dcbe01b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cb7eb8034023a23ab1e0131d9a618c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 2434, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2434, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_755c1dac97994751b74dc06b1dcbe01b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cb7eb8034023a23ab1e0131d9a618c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 2434, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2434, 4], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_c4d757f7714999dae75523d88542d3b6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.masked_select(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6069, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 6069, 4], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d2d8eb723782285c4696a43a46273d23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4d757f7714999dae75523d88542d3b6
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_d2d8eb723782285c4696a43a46273d23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4d757f7714999dae75523d88542d3b6
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069, 4], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_cdd44608b9638d77284f2c6ebe2526e9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.masked_select(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6069], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 6069], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b3c377116eb368de79b2033ed6df743f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdd44608b9638d77284f2c6ebe2526e9
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_047ff72146cf6731e6660bb10a1d2522(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.masked_select(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6069, 68], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 6069, 68], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6589bf78ce53a17b2700df4eeb9d1bfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_047ff72146cf6731e6660bb10a1d2522
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069, 68], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_d2d8eb723782285c4696a43a46273d23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4d757f7714999dae75523d88542d3b6
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_27d9574c0f15c943aca65f8fe5260423(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f62652d33ddee7084a1803608f665e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_27d9574c0f15c943aca65f8fe5260423(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f62652d33ddee7084a1803608f665e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_fad27e96c460bdf4249fbd82e2ad02f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c9b8d57a736fcff29da64d47a46c259
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_6f5d8b7a09ee8d823278a0aef0ab1cde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_034cae32af7842485cabceeecb8133ee
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 68], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_27d9574c0f15c943aca65f8fe5260423(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f62652d33ddee7084a1803608f665e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_9e7b652342e190924e506a31d866182e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.masked_select(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 11109, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 11109, 4], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6951288acb9665c3845a6aceb5bd9ba3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e7b652342e190924e506a31d866182e
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_6951288acb9665c3845a6aceb5bd9ba3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e7b652342e190924e506a31d866182e
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109, 4], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_452a38898a21555468c0f2cd999273cd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.masked_select(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 11109], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 11109], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6017d5209bb544266a209d218a71ae19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_452a38898a21555468c0f2cd999273cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_c8931f70647350ccf28a0cecc8d7fd0d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.masked_select(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 11109, 68], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 11109, 68], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_99f3055d35c9283719655e71ae5d176c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8931f70647350ccf28a0cecc8d7fd0d
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109, 68], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_6951288acb9665c3845a6aceb5bd9ba3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e7b652342e190924e506a31d866182e
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109, 4], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_e8432610be3e98c8e57568b64a44f843(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.masked_select(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2100, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 2100, 4], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_02625ab9de0fc90f994aa95d6b91e013(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8432610be3e98c8e57568b64a44f843
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_02625ab9de0fc90f994aa95d6b91e013(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8432610be3e98c8e57568b64a44f843
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100, 4], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_94b1182da0c9207227f58ea6fa7779b0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.masked_select(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2100], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 2100], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_431fda82a96d23ae728b2455eceb243d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94b1182da0c9207227f58ea6fa7779b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_2f455402808af75b75219bf5374a8697(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.masked_select(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2100, 68], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 2100, 68], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bfce7d7937b877345bc90ef237a8b04e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f455402808af75b75219bf5374a8697
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100, 68], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_02625ab9de0fc90f994aa95d6b91e013(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8432610be3e98c8e57568b64a44f843
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_96cc95f8242fe2658aa2d750b361a0d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_382bcf1a3ffc3879e89d6852a3cf27d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_96cc95f8242fe2658aa2d750b361a0d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_382bcf1a3ffc3879e89d6852a3cf27d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_d9c4c468f0a3f91d4bdcc30898c699ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c336066d6758a64b16d49d83984ac67
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_94a5d0f7b3e2bda12724050f080b91c0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.masked_select(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549, 68], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3549, 68], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1f3519957259050377326eee321a9468(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94a5d0f7b3e2bda12724050f080b91c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 68], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_96cc95f8242fe2658aa2d750b361a0d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_382bcf1a3ffc3879e89d6852a3cf27d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_8ee28deeef36cef8056dd674f8ea518b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbb4f55945237db5c3a152223b761db2
    def get_inputs(self):
        return [
            paddle.uniform([1, 500, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 500, 128], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_c9107f0d61594d8526737de948cafbdc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.masked_select(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8732, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8732, 4], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_afda62785cf2e16b79f0c5742b2a70a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9107f0d61594d8526737de948cafbdc
    def get_inputs(self):
        return [
            paddle.uniform([1, 8732, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8732, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_afda62785cf2e16b79f0c5742b2a70a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9107f0d61594d8526737de948cafbdc
    def get_inputs(self):
        return [
            paddle.uniform([1, 8732, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8732, 4], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_e3012d07d60f167807a6d35768b52fbf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.masked_select(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8400, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8400, 4], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2d5891ef505c174662c44918ca4fb6d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3012d07d60f167807a6d35768b52fbf
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_2d5891ef505c174662c44918ca4fb6d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3012d07d60f167807a6d35768b52fbf
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400, 4], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_f8a6466006d2d9dce93a8f6e61a785a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.masked_select(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8400], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8400], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_43a0a48523115d23e25be9407f5dc33d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8a6466006d2d9dce93a8f6e61a785a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_366d8fa5d1355491d587388c1eb4f3a2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.masked_select(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8400, 68], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8400, 68], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_212bef958c97cd4b3d439b4006b8c134(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_366d8fa5d1355491d587388c1eb4f3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400, 68], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_2d5891ef505c174662c44918ca4fb6d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3012d07d60f167807a6d35768b52fbf
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400, 4], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_477e79998e4fadc167a4d4bc4a755e7f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.masked_select(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9261, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9261, 4], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_86b479a035169c9a2358ea9500cbdb56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_477e79998e4fadc167a4d4bc4a755e7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_86b479a035169c9a2358ea9500cbdb56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_477e79998e4fadc167a4d4bc4a755e7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261, 4], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_bc12c755319f4ccdbce8fa3d2375efaf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.masked_select(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9261], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9261], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f91e64faefd4f82c40039e9bb15667f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc12c755319f4ccdbce8fa3d2375efaf
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_b524ed88eeaef35cc72565d8069613dc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.masked_select(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9261, 68], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9261, 68], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_18763639ca746fedcc01d585ec9a4484(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b524ed88eeaef35cc72565d8069613dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261, 68], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_86b479a035169c9a2358ea9500cbdb56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_477e79998e4fadc167a4d4bc4a755e7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261, 4], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_1a743f31b14656165eb7442cdb233019(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.masked_select(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 7581, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 7581, 4], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aefcf297721c81f744dde09869934ef3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a743f31b14656165eb7442cdb233019
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_aefcf297721c81f744dde09869934ef3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a743f31b14656165eb7442cdb233019
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581, 4], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_d335f094c54809fd3294a802f0ed4f52(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.masked_select(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 7581], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 7581], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a76ead5d324ccb8f335659c3f24e5173(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d335f094c54809fd3294a802f0ed4f52
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_ffd96d11bacc53e761cc4cd5be7a22b0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.masked_select(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 7581, 68], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 7581, 68], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9b7cd24a12d112b2403861a16613a2be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffd96d11bacc53e761cc4cd5be7a22b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581, 68], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_aefcf297721c81f744dde09869934ef3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a743f31b14656165eb7442cdb233019
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581, 4], dtype='int32'), 'bool'),
        ]




if __name__ == '__main__':
    unittest.main()