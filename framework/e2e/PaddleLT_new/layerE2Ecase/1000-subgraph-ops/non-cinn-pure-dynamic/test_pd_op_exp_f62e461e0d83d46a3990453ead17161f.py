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


class TestPrimitiveOp_3c25839756f1d5230c5b15c50c08975e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff26e98f4783c5fb3b5da69149dd25f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.2737271785736084]], [[0.27626100182533264]], [[0.1125645563006401]], [[0.006987569388002157]], [[0.13558964431285858]], [[0.052445363253355026]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_c361ad41565bd98a05e8793a68e3d408(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff26e98f4783c5fb3b5da69149dd25f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.058350659906864166]], [[0.11559354513883591]], [[0.3900896906852722]], [[0.3709835112094879]], [[0.36624351143836975]], [[0.11830221861600876]]], dtype='float32').reshape([6, 1, 1]),
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


class TestPrimitiveOp_d4efe095bc6629161dc2dea7595183ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff26e98f4783c5fb3b5da69149dd25f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.12982100248336792]], [[0.17773811519145966]], [[0.0046020327135920525]], [[0.13737168908119202]], [[0.1912841498851776]], [[0.37541282176971436]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_4bf564df57a68084242af07231e0a824(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff26e98f4783c5fb3b5da69149dd25f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.33838456869125366]], [[0.27313604950904846]], [[0.2920446991920471]], [[0.3807850778102875]], [[0.05611477419734001]], [[0.15992511808872223]]], dtype='float32').reshape([6, 1, 1]),
        ]


class PrimitiveOp_4e8cab16b6a7ae943ee5a3e23e1d4e0e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.exp(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5e929e7557db7175c31b5eef5eccdcae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e8cab16b6a7ae943ee5a3e23e1d4e0e
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.4013482332229614], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_348fe4812235207394c7d44923b3a8dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e8cab16b6a7ae943ee5a3e23e1d4e0e
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.22211085259914398], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_dfa3d157eb6624bdde73cc2f9e6f2aad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff26e98f4783c5fb3b5da69149dd25f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_65506df395b05a2633fed2249052f8c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e8cab16b6a7ae943ee5a3e23e1d4e0e
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.13775603473186493], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_133b2c8f1d7aa7308803e7d4032111ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e8cab16b6a7ae943ee5a3e23e1d4e0e
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.22658437490463257], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bdb9fc083951ca79e3f9016dea363717(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e8cab16b6a7ae943ee5a3e23e1d4e0e
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.051429517567157745], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1d5e70c791de8597b18bf94dac1e0fd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e8cab16b6a7ae943ee5a3e23e1d4e0e
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.23564143478870392], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_932dc1d22314fe27903b90ed457bfe75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e8cab16b6a7ae943ee5a3e23e1d4e0e
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.1636916697025299], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b0946e3ee3199cb8b5028b00501e56b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e8cab16b6a7ae943ee5a3e23e1d4e0e
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.09617427736520767], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6eb9a788265c6010282e2051d22614bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e8cab16b6a7ae943ee5a3e23e1d4e0e
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.21375669538974762], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b35b4cb0f93b441da0315fe8b9ac1283(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e8cab16b6a7ae943ee5a3e23e1d4e0e
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.025367464870214462], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0c41c6015d85ae96b406a1874032df41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e8cab16b6a7ae943ee5a3e23e1d4e0e
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.05307089909911156], dtype='float32').reshape([1]),
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