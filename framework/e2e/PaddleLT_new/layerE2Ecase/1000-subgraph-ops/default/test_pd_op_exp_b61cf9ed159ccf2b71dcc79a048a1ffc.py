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



class PrimitiveOp_ff26e98f4783c5fb3b5da69149dd25f5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.exp(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ab996860213e53272ce0970f388c4281(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff26e98f4783c5fb3b5da69149dd25f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.13622353971004486]], [[0.08032317459583282]], [[0.3459325134754181]], [[0.036940235644578934]], [[0.3427174985408783]], [[0.2122935652732849]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_2e5197cf6c2a8ccf4dcdde74f6bc3d67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff26e98f4783c5fb3b5da69149dd25f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.1611597239971161]], [[0.49395954608917236]], [[0.2646945118904114]], [[0.11859605461359024]], [[0.11985684931278229]], [[0.11257714033126831]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_bcd62fb094dee7f9fc5b4abf4a6acccd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff26e98f4783c5fb3b5da69149dd25f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f5a7ff01b339d69a1a7a80d9a568423d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff26e98f4783c5fb3b5da69149dd25f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fd2eb37c7ca2f58ecb934f69e196a8d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff26e98f4783c5fb3b5da69149dd25f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ff4336a116bb94ed09e400ea02628756(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff26e98f4783c5fb3b5da69149dd25f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.220103919506073]], [[0.20711898803710938]], [[0.1903029978275299]], [[0.37769195437431335]], [[0.0007872317801229656]], [[0.34398117661476135]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_4928e74afc0583dbc31d27e7a61075f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff26e98f4783c5fb3b5da69149dd25f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.2582399845123291]], [[0.27457892894744873]], [[0.43848156929016113]], [[0.03885175660252571]], [[0.03451291099190712]], [[0.36217233538627625]]], dtype='float32').reshape([6, 1, 1]),
        ]


class PrimitiveOp_e49baa4a621b00a56743bbdb0e2fe62d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.exp(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_459a684aadd28371cf9e44d4f5cdc7a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e49baa4a621b00a56743bbdb0e2fe62d
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.4943939745426178], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_09f89f7fc2568260ce0e407222d15c55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e49baa4a621b00a56743bbdb0e2fe62d
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.06618824601173401], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_dfa3d157eb6624bdde73cc2f9e6f2aad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff26e98f4783c5fb3b5da69149dd25f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c09082bc1548d309b828e9ff27e4048f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e49baa4a621b00a56743bbdb0e2fe62d
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.2636254131793976], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d741a15fa45d8c8ded4e13f0fe9be7a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e49baa4a621b00a56743bbdb0e2fe62d
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.17543905973434448], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d0de4e51d009836ceeae70aff0b9653a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e49baa4a621b00a56743bbdb0e2fe62d
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.23224054276943207], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fa29a25bd6c83c39b8e201a1df8bf339(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e49baa4a621b00a56743bbdb0e2fe62d
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.48140838742256165], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_edf521a57e8c4c315504f5d27cde74c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e49baa4a621b00a56743bbdb0e2fe62d
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.08521274477243423], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_94e124d9998288dcbbe15c8fffcef3f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e49baa4a621b00a56743bbdb0e2fe62d
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.19825448095798492], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a4fe674fa627d47aa87b617fb0a6ed69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e49baa4a621b00a56743bbdb0e2fe62d
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.2891826033592224], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d41d48503bfdf0c676527597983eedf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e49baa4a621b00a56743bbdb0e2fe62d
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.25390130281448364], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_21b00b20fe54a00c0712011249946ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e49baa4a621b00a56743bbdb0e2fe62d
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.27220115065574646], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_b4ecbf49e01c987d4a1f4b7826bc1e40(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.exp(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c019c8d4c1a7d7fb84298227a7fe100b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4ecbf49e01c987d4a1f4b7826bc1e40
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7477aa28907fe9aaa6bd4dd63e824292(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff26e98f4783c5fb3b5da69149dd25f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f7d8e388dc1a7beaf15049036c409689(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff26e98f4783c5fb3b5da69149dd25f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()