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



class PrimitiveOp_f25c23300511232bc5bdb88c3a30c452(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.logical_and(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[15200], dtype='bool'),
            paddle.static.InputSpec(shape=[15200], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f20f36267908e608e337517ca1524da6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f25c23300511232bc5bdb88c3a30c452
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[15200], dtype='int32'), 'bool'),
            paddle.cast(paddle.randint(low=0, high=2, shape=[15200], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_cf9cc351994c7b2db9c83ebc81fc73f8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.logical_and(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3800], dtype='bool'),
            paddle.static.InputSpec(shape=[3800], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_42ee603951c0542e537fa9f716a4bcae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf9cc351994c7b2db9c83ebc81fc73f8
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
            paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_42ee603951c0542e537fa9f716a4bcae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf9cc351994c7b2db9c83ebc81fc73f8
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
            paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_455a3cfbd1dfd524bd5462fa89270ec9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.logical_and(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2204], dtype='bool'),
            paddle.static.InputSpec(shape=[2204], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a475231370c7564cf6d23fd89dd30eaf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_455a3cfbd1dfd524bd5462fa89270ec9
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[2204], dtype='int32'), 'bool'),
            paddle.cast(paddle.randint(low=0, high=2, shape=[2204], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_5e1bff65b5173c8f9b0919cc2cac1afe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.logical_and(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[950], dtype='bool'),
            paddle.static.InputSpec(shape=[950], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f4cb5e674c7b817f62e3f79d0f1e38ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e1bff65b5173c8f9b0919cc2cac1afe
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[950], dtype='int32'), 'bool'),
            paddle.cast(paddle.randint(low=0, high=2, shape=[950], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_f20f36267908e608e337517ca1524da6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f25c23300511232bc5bdb88c3a30c452
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[15200], dtype='int32'), 'bool'),
            paddle.cast(paddle.randint(low=0, high=2, shape=[15200], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_cb78c7f4ecbff75705aa64b9dde27ac0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.logical_and(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8816], dtype='bool'),
            paddle.static.InputSpec(shape=[8816], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1311990f53a3df9b7e827d24a29505cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb78c7f4ecbff75705aa64b9dde27ac0
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[8816], dtype='int32'), 'bool'),
            paddle.cast(paddle.randint(low=0, high=2, shape=[8816], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_4379f061becfaae42e82118d3b79f041(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.logical_and(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[150], dtype='bool'),
            paddle.static.InputSpec(shape=[150], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3958d36e330db35c2ee6c2a09df22c2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4379f061becfaae42e82118d3b79f041
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[150], dtype='int32'), 'bool'),
            paddle.cast(paddle.randint(low=0, high=2, shape=[150], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_40bbc07f057214b0a72935473b99d78a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.logical_and(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[70], dtype='bool'),
            paddle.static.InputSpec(shape=[70], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_91831e4f19ae0de1df40db8d44bb02a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40bbc07f057214b0a72935473b99d78a
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[70], dtype='int32'), 'bool'),
            paddle.cast(paddle.randint(low=0, high=2, shape=[70], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_f4cb5e674c7b817f62e3f79d0f1e38ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e1bff65b5173c8f9b0919cc2cac1afe
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[950], dtype='int32'), 'bool'),
            paddle.cast(paddle.randint(low=0, high=2, shape=[950], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_91831e4f19ae0de1df40db8d44bb02a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40bbc07f057214b0a72935473b99d78a
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[70], dtype='int32'), 'bool'),
            paddle.cast(paddle.randint(low=0, high=2, shape=[70], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_eb9fa03104c0d88f5671352f189a5dad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.logical_and(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[551], dtype='bool'),
            paddle.static.InputSpec(shape=[551], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3452f4b1ec2fe0cabc72b52ae1faf542(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb9fa03104c0d88f5671352f189a5dad
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[551], dtype='int32'), 'bool'),
            paddle.cast(paddle.randint(low=0, high=2, shape=[551], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_42ee603951c0542e537fa9f716a4bcae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf9cc351994c7b2db9c83ebc81fc73f8
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
            paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_3760f369f61f3ceba7e30e358ff13ed6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.logical_and(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[247], dtype='bool'),
            paddle.static.InputSpec(shape=[247], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fcb7b8c4c56d72072eb7fd159e382e96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3760f369f61f3ceba7e30e358ff13ed6
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[247], dtype='int32'), 'bool'),
            paddle.cast(paddle.randint(low=0, high=2, shape=[247], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_fcb7b8c4c56d72072eb7fd159e382e96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3760f369f61f3ceba7e30e358ff13ed6
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[247], dtype='int32'), 'bool'),
            paddle.cast(paddle.randint(low=0, high=2, shape=[247], dtype='int32'), 'bool'),
        ]




if __name__ == '__main__':
    unittest.main()