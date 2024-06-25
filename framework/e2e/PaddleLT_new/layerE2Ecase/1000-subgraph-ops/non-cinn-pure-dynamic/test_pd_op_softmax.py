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



class PrimitiveOp_1a42924ded9940b812e73909b2f36b7f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3e98a64b48315f15ef209ec327221ce6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 1, 91], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5bce6db46f6e283211e79987efc4c9be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 100], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_34b139cda56b62da36381f86033c4e22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 198, 198], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4d5755df31e3819ec16c1c5b8ad33fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a4d84455a622d0fd2834fba6ff96f11f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 320], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8c1899da69bca14c7a58eb96aa59f630(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_78a2996f570cb2b2b22431b5028cafdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c1899da69bca14c7a58eb96aa59f630
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 32768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_96c3b21813daa4ef97691586c5880bcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a31e306055f48902b2ab7cf75768e9c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c1899da69bca14c7a58eb96aa59f630
    def get_inputs(self):
        return [
            paddle.uniform([1, 21, 16384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_22e47409bd09b7de9eaf0a0652b4bd71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10aec829105e988405d3826b6b8657d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 577, 577], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_86ef9b14a6c72ff50bdc370b9b517792(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c45f7a0e009c14fd2e5465c590e619fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36ac6c6d85016a2b898bdf97d8e20b96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fb9e6805effb4180e1f77d3ddc777ffe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 640, 640], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_09fb8133fd2328477d2440ee24ed23f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 198, 198], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b38ca55d71d5fb117d431679b5e5dee4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5bce6db46f6e283211e79987efc4c9be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 100], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f626a56e3c7a1f2b0edab865e9f539f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a389c97e533c70cdc91d73a3aa446add(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_92bef81574dd68062d19494c45e77c21(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fca0c0d6760880c7eeee362925e88499(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92bef81574dd68062d19494c45e77c21
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 13, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dbea4ba5dba209f733bfbc02a2afc743(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1025, 1025], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_204cf2219ba7ff677e1f66eaa5320910(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.softmax(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c532171d14420ba62f8504b5c6f7f6c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cf2219ba7ff677e1f66eaa5320910
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_56be237ef966ceec823048b23ad621c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c625c1c4924d3bbbecd177e1344b7302(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0256dff114f6523536d50569d5c0e54a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 197, 197], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e21a9b005fab6b7e7dacaef658b86935(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a4d84455a622d0fd2834fba6ff96f11f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 320], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e9e0d661c45c4d0df792486a252801ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7ec1b084052cf076310878b65f4237bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 200, 200], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9dbe7a0ec3708de4698325405f306bd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_472707bf0f0a9009dbb5a0dce674c02c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_56be237ef966ceec823048b23ad621c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c20b3bd1496eb12724b3e94a9e83cf5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_adb8b168d1b12b1d66504d5662e32032(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cf2219ba7ff677e1f66eaa5320910
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 8192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_15c0e0cd313a5d017431410db4b975da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1025, 1025], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3da364a8c5569ee93c39de2b9d44b94b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cf2219ba7ff677e1f66eaa5320910
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c625c1c4924d3bbbecd177e1344b7302(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_03950f45f7e0b73f4031772f612d9ccf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fc21e79212501a4293ddfcfa19781cfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92bef81574dd68062d19494c45e77c21
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 50, 76], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7bd9a5faf6fa727c09331813b100e0d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7bd9a5faf6fa727c09331813b100e0d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b38ca55d71d5fb117d431679b5e5dee4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_86ef9b14a6c72ff50bdc370b9b517792(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_472707bf0f0a9009dbb5a0dce674c02c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fd4611c84bf28761a80516c98d3bee52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 197, 197], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e9e0d661c45c4d0df792486a252801ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3da364a8c5569ee93c39de2b9d44b94b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cf2219ba7ff677e1f66eaa5320910
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd16dee023d471c4a50103d8661e3a34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_14afa30db663f3713d1e8c1cfd8c4f33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 160, 160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c9a11a6a7340c1a3952174599158dff8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1174, 1174], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a771b4f5a4a2fe324de8f8d9774f7bc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92bef81574dd68062d19494c45e77c21
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 25, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_88f3a067e2c5f314fc2974fbde826332(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92bef81574dd68062d19494c45e77c21
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 7, 10], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1ebc5f48400d9d74df893d0ae5f38a9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1174, 1174], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e21a9b005fab6b7e7dacaef658b86935(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_76da0a81dc6061673dcdaf1885f15905(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92bef81574dd68062d19494c45e77c21
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 100, 152], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd48d865600bec06ba2e9856ac6ba83a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a42924ded9940b812e73909b2f36b7f
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 50, 50], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()