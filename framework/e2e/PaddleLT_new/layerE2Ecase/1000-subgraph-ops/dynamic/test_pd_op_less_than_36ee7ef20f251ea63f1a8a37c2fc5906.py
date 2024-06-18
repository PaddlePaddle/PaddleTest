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



class PrimitiveOp_488dbad4033121c15e51966b4818934c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 < input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d5a6c679cfc9c3c046018f47e5f0f464(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_488dbad4033121c15e51966b4818934c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[15200], dtype='int64'),
            paddle.to_tensor(80, dtype='int64').reshape([]),
        ]


class PrimitiveOp_851abb00a2070fc601fb680880ea59e0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 < input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ae99f8b01a25fc7a25696afb31c76f89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_851abb00a2070fc601fb680880ea59e0
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_fe524ea36876e1fc7c39330a4dbfee89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_488dbad4033121c15e51966b4818934c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[3800], dtype='int64'),
            paddle.to_tensor(80, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_fe524ea36876e1fc7c39330a4dbfee89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_488dbad4033121c15e51966b4818934c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[3800], dtype='int64'),
            paddle.to_tensor(80, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_d7c9f0b1fa587f6b8793906a710a7267(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_488dbad4033121c15e51966b4818934c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2204], dtype='int64'),
            paddle.to_tensor(80, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_558c4edfbbc8b07b8bff0c575f7c0914(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_488dbad4033121c15e51966b4818934c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[950], dtype='int64'),
            paddle.to_tensor(80, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_cf4859d5c1b5b31e24f3ffa06f25c7d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_488dbad4033121c15e51966b4818934c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[15200], dtype='int64'),
            paddle.to_tensor(81, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_533a84dd2cb61b3e3e99a13b4e1164f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_488dbad4033121c15e51966b4818934c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[8816], dtype='int64'),
            paddle.to_tensor(80, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_e1ce4665cf1b796d19a7ed848560a3a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_488dbad4033121c15e51966b4818934c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[150], dtype='int64'),
            paddle.to_tensor(80, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_ed23c63b3126b2c0dcd45a4bbb408aca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_488dbad4033121c15e51966b4818934c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[70], dtype='int64'),
            paddle.to_tensor(80, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_2726f116659301595cc6b8e0eee4add9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_851abb00a2070fc601fb680880ea59e0
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3f2eee261c0689fb68d3e913184f789e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_488dbad4033121c15e51966b4818934c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[950], dtype='int64'),
            paddle.to_tensor(81, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_e5e183811fbd8c85f88ab6fbfd2f0c30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_851abb00a2070fc601fb680880ea59e0
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c5512255e2eaa8692b6ebc5e9338553b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_488dbad4033121c15e51966b4818934c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[70], dtype='int64'),
            paddle.to_tensor(81, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_c4c9e23bbeed9f340207c97974471a60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_488dbad4033121c15e51966b4818934c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[551], dtype='int64'),
            paddle.to_tensor(80, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_d785ec23f9c8159a921462b5d9971671(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_851abb00a2070fc601fb680880ea59e0
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_11cb9ea0dfe9972f37f2875f7832b041(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_488dbad4033121c15e51966b4818934c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[3800], dtype='int64'),
            paddle.to_tensor(81, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_e0bba86bb8b13c4d5ac662ccffb25894(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_851abb00a2070fc601fb680880ea59e0
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e659e5041bddb1dd72e19952cf647edc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_488dbad4033121c15e51966b4818934c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[247], dtype='int64'),
            paddle.to_tensor(80, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_54f1ab6f7ac31641424226456c16a7af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_488dbad4033121c15e51966b4818934c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[247], dtype='int64'),
            paddle.to_tensor(81, dtype='int64').reshape([]),
        ]




if __name__ == '__main__':
    unittest.main()