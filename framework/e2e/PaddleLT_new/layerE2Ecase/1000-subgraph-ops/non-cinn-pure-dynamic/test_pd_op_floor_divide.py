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



class PrimitiveOp_eccfe54bb7aa3c9a1b04798df2044a55(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.floor_divide(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4e2cd8dc7b737167081d49c4f22f4edb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eccfe54bb7aa3c9a1b04798df2044a55
    def get_inputs(self):
        return [
            paddle.to_tensor(528, dtype='int32').reshape([]),
            paddle.to_tensor(24, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_8d47947141fd65d9e68735075ebd013b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eccfe54bb7aa3c9a1b04798df2044a55
    def get_inputs(self):
        return [
            paddle.to_tensor(12, dtype='int32').reshape([]),
            paddle.to_tensor(2, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_9f404fc0adc802ff530094b6a504f64b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eccfe54bb7aa3c9a1b04798df2044a55
    def get_inputs(self):
        return [
            paddle.to_tensor(384, dtype='int32').reshape([]),
            paddle.to_tensor(96, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_fbe370fefc87df4991f7e6e2e1e4a24a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eccfe54bb7aa3c9a1b04798df2044a55
    def get_inputs(self):
        return [
            paddle.to_tensor(20, dtype='int32').reshape([]),
            paddle.to_tensor(2, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_3df1bfaf40e558536eaa1aee5305f3b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eccfe54bb7aa3c9a1b04798df2044a55
    def get_inputs(self):
        return [
            paddle.to_tensor(2, dtype='int32').reshape([]),
            paddle.to_tensor(2, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_f343e990a71d1ee4d9d8e2c8af1582dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eccfe54bb7aa3c9a1b04798df2044a55
    def get_inputs(self):
        return [
            paddle.to_tensor(4, dtype='int32').reshape([]),
            paddle.to_tensor(2, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_4238685d005289ad7f83578680641d35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eccfe54bb7aa3c9a1b04798df2044a55
    def get_inputs(self):
        return [
            paddle.to_tensor(576, dtype='int32').reshape([]),
            paddle.to_tensor(96, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_1271474f4dc0d03c9d9cceb544c12f26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eccfe54bb7aa3c9a1b04798df2044a55
    def get_inputs(self):
        return [
            paddle.to_tensor(96, dtype='int32').reshape([]),
            paddle.to_tensor(24, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_8d47947141fd65d9e68735075ebd013b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eccfe54bb7aa3c9a1b04798df2044a55
    def get_inputs(self):
        return [
            paddle.to_tensor(12, dtype='int32').reshape([]),
            paddle.to_tensor(2, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_503f92b3007f289a532f5a45ae526e21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eccfe54bb7aa3c9a1b04798df2044a55
    def get_inputs(self):
        return [
            paddle.to_tensor(960, dtype='int32').reshape([]),
            paddle.to_tensor(96, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_861cbd09900bb679e026c674913c1dd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eccfe54bb7aa3c9a1b04798df2044a55
    def get_inputs(self):
        return [
            paddle.to_tensor(2112, dtype='int32').reshape([]),
            paddle.to_tensor(96, dtype='int32').reshape([]),
        ]


class PrimitiveOp_a17ea61abf8e603001ac81207f789325(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.floor_divide(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_61c7fe9bf057d02974a8787d7c5b81f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a17ea61abf8e603001ac81207f789325
    def get_inputs(self):
        return [
            paddle.to_tensor([4], dtype='int32').reshape([1]),
            paddle.to_tensor(2, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_81b15cba5a1f4ba1339a71228d58033d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a17ea61abf8e603001ac81207f789325
    def get_inputs(self):
        return [
            paddle.to_tensor([7], dtype='int32').reshape([1]),
            paddle.to_tensor(2, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_d1e403f5a00a1c983d58da0766fb1e62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eccfe54bb7aa3c9a1b04798df2044a55
    def get_inputs(self):
        return [
            paddle.to_tensor(8, dtype='int32').reshape([]),
            paddle.to_tensor(2, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_a94e8d4ffa9d7e2cac84b2c6eae6eb14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eccfe54bb7aa3c9a1b04798df2044a55
    def get_inputs(self):
        return [
            paddle.to_tensor(240, dtype='int32').reshape([]),
            paddle.to_tensor(24, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_e5e5cd34e7bd51827385880d7d14f892(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eccfe54bb7aa3c9a1b04798df2044a55
    def get_inputs(self):
        return [
            paddle.to_tensor(44, dtype='int32').reshape([]),
            paddle.to_tensor(2, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_bc2c101e20e49ba346f137dd89eca7a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a17ea61abf8e603001ac81207f789325
    def get_inputs(self):
        return [
            paddle.to_tensor([28], dtype='int32').reshape([1]),
            paddle.to_tensor(7, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_ad9a4ed6d9a5b290f0516e609ac5c9ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a17ea61abf8e603001ac81207f789325
    def get_inputs(self):
        return [
            paddle.to_tensor([77], dtype='int32').reshape([1]),
            paddle.to_tensor(7, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_f1f43c43ceb4160e9f7d71d0dc6aa3c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eccfe54bb7aa3c9a1b04798df2044a55
    def get_inputs(self):
        return [
            paddle.to_tensor(144, dtype='int32').reshape([]),
            paddle.to_tensor(24, dtype='int32').reshape([]),
        ]




if __name__ == '__main__':
    unittest.main()