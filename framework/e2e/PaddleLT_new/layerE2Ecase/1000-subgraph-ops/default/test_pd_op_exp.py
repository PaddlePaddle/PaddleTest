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


class TestPrimitiveOp_d2c162603ad0689fb684612643868c43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff26e98f4783c5fb3b5da69149dd25f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.1779407411813736]], [[0.21024881303310394]], [[0.4890906512737274]], [[0.3003374934196472]], [[0.36084774136543274]], [[0.4671012759208679]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_f73d41adc27004369ee96e5b7236b687(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff26e98f4783c5fb3b5da69149dd25f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.09420384466648102]], [[0.4809618890285492]], [[0.04842865839600563]], [[0.32863229513168335]], [[0.21174632012844086]], [[0.3063952326774597]]], dtype='float32').reshape([6, 1, 1]),
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


class TestPrimitiveOp_9a5a55cc96d558b81a9a08322b1761d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff26e98f4783c5fb3b5da69149dd25f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.10242156684398651]], [[0.23851335048675537]], [[0.3752126693725586]], [[0.019431715831160545]], [[0.34421154856681824]], [[0.24010156095027924]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_6e78e9935a714bd94d5d28980afcd386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff26e98f4783c5fb3b5da69149dd25f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.02652212604880333]], [[0.28697821497917175]], [[0.2873174548149109]], [[0.231895312666893]], [[0.3593861162662506]], [[0.14348183572292328]]], dtype='float32').reshape([6, 1, 1]),
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


class TestPrimitiveOp_267e80e395db131a1d73f457abb58dd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e49baa4a621b00a56743bbdb0e2fe62d
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.07361782342195511], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6eff97b4687575cd738429c04eb5a90f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e49baa4a621b00a56743bbdb0e2fe62d
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.14041727781295776], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_dfa3d157eb6624bdde73cc2f9e6f2aad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff26e98f4783c5fb3b5da69149dd25f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c64fc402fdc94a12446fac978f8ac043(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e49baa4a621b00a56743bbdb0e2fe62d
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.4471670091152191], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_53b6fd9b3a5127869f229ed8c371a284(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e49baa4a621b00a56743bbdb0e2fe62d
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.18567442893981934], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8a3b7aaa6b71fcf9dac41f922a709a12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e49baa4a621b00a56743bbdb0e2fe62d
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.3839329481124878], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c5aa6fa44de652441a894dc37d1ce438(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e49baa4a621b00a56743bbdb0e2fe62d
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.10538749396800995], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5c700a89fcc3663e0e0b4820d78b1dfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e49baa4a621b00a56743bbdb0e2fe62d
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.3118523955345154], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fe8847f1452ca724c63b3247c3b5e6ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e49baa4a621b00a56743bbdb0e2fe62d
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.20082803070545197], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7f54e4116a0d15f9172ca3ae01edea82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e49baa4a621b00a56743bbdb0e2fe62d
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.04535200446844101], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ec7aeff79f88d214eeae060a050f0730(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e49baa4a621b00a56743bbdb0e2fe62d
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.3101944327354431], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e2b28850ed5d59bfbfde38a165eb388(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e49baa4a621b00a56743bbdb0e2fe62d
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.2691633105278015], dtype='float32').reshape([1]),
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