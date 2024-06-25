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



class PrimitiveOp_3809c0dd614b2274b82d62b68c761ce0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.bmm(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 19, 32768], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 32768, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7bb512fb04e93c7c2351513e13c00246(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3809c0dd614b2274b82d62b68c761ce0
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 32768], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32768, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cd48d639335ed9db1aaa0a5af9b532b4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.bmm(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 21, 16384], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 16384, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5b401276c3bb078ab02754d68da1bde6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd48d639335ed9db1aaa0a5af9b532b4
    def get_inputs(self):
        return [
            paddle.uniform([1, 21, 16384], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 16384, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4efff31152d5552b155ff41859dc9d2e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.bmm(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4096, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 64, 4096], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7283a350fd35241687a4d29cca9ef813(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4efff31152d5552b155ff41859dc9d2e
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 4096], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_19dd6c29141d3d7423e14ec5445f7b11(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.bmm(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 4096], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4096, 4096], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2811b4a6fd01473517a83fee42f4430c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19dd6c29141d3d7423e14ec5445f7b11
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 4096], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4096, 4096], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_19a35f9a5ff6031629bfeb6ee627d8d1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.bmm(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8192, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 64, 8192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c00bdb44f90dd390c5ff4d37a382c6bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19a35f9a5ff6031629bfeb6ee627d8d1
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 8192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e1e441a8287883dd286b7f7348b242cc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.bmm(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 8192], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8192, 8192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_791728b81af3cb22b08a946c60746aef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1e441a8287883dd286b7f7348b242cc
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 8192], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8192, 8192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f25ff22d7c0eb0dc5cbed1af0d7cc6d8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.bmm(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 8192], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8192, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_393794fe57f37d657fa2586e0998aa6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f25ff22d7c0eb0dc5cbed1af0d7cc6d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 8192], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8192, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_23399bcd9ac30a7e0f934fd6be7a6dc8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.bmm(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 512, 8192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f22b2ea51a9ddbd6d70da2de6f5c066a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23399bcd9ac30a7e0f934fd6be7a6dc8
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 8192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_65adffb5a63e2e19d0051b65aaf35c8c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.bmm(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 4096], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4096, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b303c39bb1001f846fa9721f22390edc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65adffb5a63e2e19d0051b65aaf35c8c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 4096], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4096, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ac9439f92d5d83542cc995831db465b3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.bmm(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 512, 4096], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eaf477bd42b78c4afb250badefeea684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac9439f92d5d83542cc995831db465b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 4096], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()