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



class PrimitiveOp_52218334a33ce0277324a8b578a63d85(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.elu(input_0, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 21504, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d1db6ca6f42f66bafb60f1551c99af4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_52218334a33ce0277324a8b578a63d85
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 2], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_16ab534a9a1e1659ee5b25aec9bba1ac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.elu(input_0, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 8, 8], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_26c0511ac9c820b5f628dff6908f060a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16ab534a9a1e1659ee5b25aec9bba1ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_af6bec848d7d5955a1c0dadd67767c3b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.elu(input_0, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_22f0795422a61d25ea9c944f415272ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af6bec848d7d5955a1c0dadd67767c3b
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f23f640443b180128cf4099fa9224ebe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.elu(input_0, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 128, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fda293814df9ac1a5fbd7f498023cef8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f23f640443b180128cf4099fa9224ebe
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5d8de80f56be9d32bde122b3c34207c7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.elu(input_0, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 16, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8ccf10a8a4d651ba809a84510708095a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d8de80f56be9d32bde122b3c34207c7
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_13283540e8da948da1c7b2b29d4ec6ce(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.elu(input_0, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 32, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8184c29cd5b14ad4c1eebaa754649c50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13283540e8da948da1c7b2b29d4ec6ce
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()